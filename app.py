"""
app.py — Driver Drowsiness Detection Backend
FastAPI + WebSocket + ONNX + MediaPipe + LSTM
Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import cv2, numpy as np, onnxruntime as ort, asyncio, base64, json, torch
import torch.nn as nn, urllib.request, os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial.distance import euclidean
from collections import deque
from pathlib import Path

# ── MediaPipe (new API) ───────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

MODEL_PATH = './face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print('Downloading face landmark model...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        MODEL_PATH
    )

options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = FaceLandmarker.create_from_options(options)

class LM:
    def __init__(self, x, y, z): self.x=x; self.y=y; self.z=z

def get_landmarks(bgr_frame):
    rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    return [LM(lm.x, lm.y, lm.z) for lm in result.face_landmarks[0]]

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE       = [362, 385, 387, 263, 373, 380]
RIGHT_EYE      = [33,  160, 158, 133, 153, 144]
MOUTH          = [61, 291, 39, 181, 0, 17, 269, 405]
POSE_LANDMARKS = [1, 152, 226, 446, 57, 287]
MODEL_3D       = np.array([
    [0.0,    0.0,    0.0  ], [0.0,   -330.0, -65.0 ],
    [-225.0, 170.0, -135.0], [225.0,  170.0, -135.0],
    [-150.0,-150.0, -125.0], [150.0, -150.0, -125.0],
], dtype=np.float64)

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESH, MAR_THRESH = 0.20, 0.60
PITCH_THRESH, YAW_THRESH = 15.0, 30.0
W_CNN, W_EAR, W_MAR, W_POSE = 0.40, 0.30, 0.15, 0.15
IMG_SIZE = 96
MEAN = np.array([0.485,0.456,0.406],dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.229,0.224,0.225],dtype=np.float32).reshape(3,1,1)

# ── Signal functions ──────────────────────────────────────────────────────────
def eye_aspect_ratio(lm, idx, w, h):
    p = np.array([[lm[i].x*w, lm[i].y*h] for i in idx])
    return (euclidean(p[1],p[5])+euclidean(p[2],p[4]))/(2*euclidean(p[0],p[3])+1e-6)

def mouth_aspect_ratio(lm, idx, w, h):
    p = np.array([[lm[i].x*w, lm[i].y*h] for i in idx])
    return ((euclidean(p[2],p[6])+euclidean(p[3],p[7]))/2)/(euclidean(p[0],p[1])+1e-6)

def head_pose_angles(lm, idx, w, h):
    ip = np.array([[lm[i].x*w, lm[i].y*h] for i in idx], dtype=np.float64)
    cm = np.array([[float(w),0,w/2],[0,float(w),h/2],[0,0,1]],dtype=np.float64)
    ok,rv,_ = cv2.solvePnP(MODEL_3D,ip,cm,np.zeros((4,1)),flags=cv2.SOLVEPNP_SQPNP)
    if not ok: return 0.,0.,0.
    rm,_ = cv2.Rodrigues(rv)
    a,*_ = cv2.RQDecomp3x3(rm)
    return a[0], a[1], a[2]

# ── ONNX ──────────────────────────────────────────────────────────────────────
sess       = ort.InferenceSession('./drowsiness.onnx', providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

def cnn_drowsy_prob(crop):
    img = cv2.resize(crop,(IMG_SIZE,IMG_SIZE))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    img = (img.transpose(2,0,1)-MEAN)/STD
    logits = sess.run(None,{input_name: img[np.newaxis]})[0][0]
    p = np.exp(logits)/np.exp(logits).sum()
    return float(p[0])

# ── LSTM ──────────────────────────────────────────────────────────────────────
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 30

class DrowsinessLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size,32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32,2)
        )
    def forward(self, x):
        _, (hn,_) = self.lstm(x)
        return self.classifier(hn[-1])

lstm_model = DrowsinessLSTM().to(DEVICE)
if os.path.exists('./lstm_model.pth'):
    lstm_model.load_state_dict(torch.load('./lstm_model.pth', map_location=DEVICE))
    lstm_model.eval()
    print('✅ LSTM loaded')
else:
    print('⚠️  lstm_model.pth not found — LSTM score will be skipped')

# ── Per-session state ─────────────────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.buffer          = deque(maxlen=SEQ_LEN)
        self.ear_samples     = []
        self.baseline_ear    = None
        self.personal_thresh = None
        self.calibrated      = False
        self.score_history   = deque(maxlen=150)  # last 5 seconds at 30fps

    def calibrate(self, ear):
        if self.calibrated: return
        self.ear_samples.append(ear)
        if len(self.ear_samples) >= 30:
            self.baseline_ear    = float(np.mean(self.ear_samples))
            self.personal_thresh = self.baseline_ear * 0.65
            self.calibrated      = True

    @property
    def ear_thresh(self):
        return self.personal_thresh if self.calibrated else EAR_THRESH

    @property
    def calib_progress(self):
        return min(len(self.ear_samples) / 30, 1.0)

# ── Process one frame ─────────────────────────────────────────────────────────
def process_frame(bgr, state: SessionState):
    h, w = bgr.shape[:2]
    lm   = get_landmarks(bgr)

    if lm is None:
        return {'error': 'no_face', 'calibrated': state.calibrated,
                'calib_progress': state.calib_progress}

    # Signals
    ear   = (eye_aspect_ratio(lm,LEFT_EYE,w,h) + eye_aspect_ratio(lm,RIGHT_EYE,w,h)) / 2
    mar   = mouth_aspect_ratio(lm, MOUTH, w, h)
    pitch, yaw, _ = head_pose_angles(lm, POSE_LANDMARKS, w, h)

    state.calibrate(ear)
    thresh = state.ear_thresh

    xs=[lm[i].x*w for i in range(468)]; ys=[lm[i].y*h for i in range(468)]
    x1,x2=max(0,int(min(xs))-10),min(w,int(max(xs))+10)
    y1,y2=max(0,int(min(ys))-10),min(h,int(max(ys))+10)
    crop = bgr[y1:y2, x1:x2]
    cnn  = cnn_drowsy_prob(crop) if crop.size > 0 else 0.5

    # Fusion score
    ear_s  = float(np.clip(1-(ear/thresh),0,1))
    mar_s  = float(np.clip(mar/MAR_THRESH,0,1))
    pose_s = float(np.clip(max(abs(pitch)/PITCH_THRESH,abs(yaw)/YAW_THRESH),0,1))
    fusion = W_CNN*cnn + W_EAR*ear_s + W_MAR*mar_s + W_POSE*pose_s

    # LSTM score
    feat = [cnn,
            float(np.clip(ear/0.4,0,1)),
            float(np.clip(mar,0,1)),
            float(np.clip((pitch+30)/60,0,1)),
            float(np.clip((yaw+45)/90,0,1)),
            0.0]
    state.buffer.append(feat)

    lstm_score = None
    if len(state.buffer) == SEQ_LEN and os.path.exists('./lstm_model.pth'):
        seq = torch.FloatTensor(np.array(state.buffer)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(lstm_model(seq), dim=1)[0]
        lstm_score = float(probs[1])

    # Final score — blend fusion + LSTM if available
    final = (0.5*fusion + 0.5*lstm_score) if lstm_score is not None else fusion
    state.score_history.append(final)

    # PERCLOS
    perclos = sum(1 for s in state.score_history if s > 0.65) / max(len(state.score_history),1)

    if final >= 0.70:   alert = 'ALERT'
    elif final >= 0.50: alert = 'WARNING'
    else:               alert = 'AWAKE'

    return {
        'alert':        alert,
        'final_score':  round(final, 3),
        'fusion_score': round(fusion, 3),
        'lstm_score':   round(lstm_score, 3) if lstm_score else None,
        'ear':          round(ear, 3),
        'mar':          round(mar, 3),
        'pitch':        round(pitch, 2),
        'yaw':          round(yaw, 2),
        'cnn':          round(cnn, 3),
        'perclos':      round(perclos, 3),
        'calibrated':   state.calibrated,
        'calib_progress': round(state.calib_progress, 2),
        'ear_thresh':   round(thresh, 3),
        'history':      list(state.score_history)[-60:],  # last 2 seconds
    }

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title='Driver Drowsiness Detection')
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])
app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/')
async def root():
    return HTMLResponse(Path('templates/index.html').read_text())

@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = SessionState()
    print('Client connected')

    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)

            if msg.get('type') == 'frame':
                # Decode base64 frame from browser
                img_data = base64.b64decode(msg['data'].split(',')[1])
                nparr    = np.frombuffer(img_data, np.uint8)
                bgr      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if bgr is not None:
                    result = process_frame(bgr, state)
                    await ws.send_text(json.dumps(result))

            elif msg.get('type') == 'reset_calibration':
                state.ear_samples     = []
                state.baseline_ear    = None
                state.personal_thresh = None
                state.calibrated      = False
                await ws.send_text(json.dumps({'type': 'calibration_reset'}))

    except WebSocketDisconnect:
        print('Client disconnected')
