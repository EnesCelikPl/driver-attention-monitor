import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import time
import subprocess
import math

model_path = str(Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

EAR_DROWSY_THRESHOLD = 0.20
EYE_CLOSED_SECONDS = 1.5

SQUINT_EAR_MIN = 0.21
SQUINT_EAR_MAX = 0.26
SQUINT_SECONDS = 2.0
SQUINT_HOLD_SECONDS = 2.0

YAWN_THRESHOLD = 0.55
YAWN_MIN_SECONDS = 0.8
YAWN_HOLD_SECONDS = 5.0
YAWN_WINDOW_SECONDS = 300.0
YAWN_COUNT_TRIGGER = 3
YAWN_COOLDOWN_SECONDS = 2.0
YAWN_DROWSY_COOLDOWN_SECONDS = 60.0

DROWSY_HOLD_SECONDS = 3.0

HEAD_DOWN_PITCH_DEG = 18.0
LOOK_AWAY_YAW_DEG = 25.0
POSE_SECONDS = 2.0
POSE_HOLD_SECONDS = 2.5

BEEP_INTERVAL_SECONDS = 0.65

CALIB_SECONDS = 4.0

closed_start = None
squint_start = None
yawn_start = None
pose_start = None

yawn_until = 0.0
squint_until = 0.0
drowsy_until = 0.0
pose_until = 0.0

last_yawn_time = 0.0
last_yawn_drowsy_trigger = 0.0
yawn_events = []

was_drowsy = False
muted = False
last_beep_time = 0.0

last_fps_time = time.time()
fps = 0.0

calib_start = time.time()
calib_values = []
calibrated = False
baseline_ear = None
dyn_squint_min = SQUINT_EAR_MIN
dyn_squint_max = SQUINT_EAR_MAX
dyn_drowsy_thr = EAR_DROWSY_THRESHOLD

def dist(a, b):
    return np.linalg.norm(a - b)

def ear_one(landmarks, idx):
    p1 = np.array([landmarks[idx[0]].x, landmarks[idx[0]].y])
    p2 = np.array([landmarks[idx[1]].x, landmarks[idx[1]].y])
    p3 = np.array([landmarks[idx[2]].x, landmarks[idx[2]].y])
    p4 = np.array([landmarks[idx[3]].x, landmarks[idx[3]].y])
    p5 = np.array([landmarks[idx[4]].x, landmarks[idx[4]].y])
    p6 = np.array([landmarks[idx[5]].x, landmarks[idx[5]].y])
    v1 = dist(p2, p6)
    v2 = dist(p3, p5)
    h = dist(p1, p4)
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def ear_avg(landmarks):
    return (ear_one(landmarks, LEFT_EYE) + ear_one(landmarks, RIGHT_EYE)) / 2.0

def mar(landmarks):
    top = np.array([landmarks[MOUTH_TOP].x, landmarks[MOUTH_TOP].y])
    bottom = np.array([landmarks[MOUTH_BOTTOM].x, landmarks[MOUTH_BOTTOM].y])
    left = np.array([landmarks[MOUTH_LEFT].x, landmarks[MOUTH_LEFT].y])
    right = np.array([landmarks[MOUTH_RIGHT].x, landmarks[MOUTH_RIGHT].y])
    v = dist(top, bottom)
    h = dist(left, right)
    if h == 0:
        return 0.0
    return v / h

def draw_right(frame, text, y, scale=0.8, thickness=2):
    h, w = frame.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = w - tw - 20
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)

def beep():
    try:
        subprocess.Popen(["osascript", "-e", "beep 1"])
    except Exception:
        pass

def to_deg(rad):
    return rad * 180.0 / math.pi

def head_pose_deg(frame, landmarks):
    h, w = frame.shape[:2]
    idx = {"nose": 1, "chin": 152, "l_eye": 33, "r_eye": 263, "l_mouth": 61, "r_mouth": 291}

    image_points = np.array([
        [landmarks[idx["nose"]].x * w, landmarks[idx["nose"]].y * h],
        [landmarks[idx["chin"]].x * w, landmarks[idx["chin"]].y * h],
        [landmarks[idx["l_eye"]].x * w, landmarks[idx["l_eye"]].y * h],
        [landmarks[idx["r_eye"]].x * w, landmarks[idx["r_eye"]].y * h],
        [landmarks[idx["l_mouth"]].x * w, landmarks[idx["l_mouth"]].y * h],
        [landmarks[idx["r_mouth"]].x * w, landmarks[idx["r_mouth"]].y * h],
    ], dtype=np.float64)

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0

    return to_deg(x), to_deg(y), to_deg(z)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    now = time.time()

    dt = now - last_fps_time
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
    last_fps_time = now

    yawn_events = [t for t in yawn_events if now - t <= YAWN_WINDOW_SECONDS]
    yawns_5m = len(yawn_events)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    ear_value = None
    mar_value = None
    pitch = None
    yaw = None
    roll = None

    if result.face_landmarks:
        lms = result.face_landmarks[0]

        ear_value = ear_avg(lms)
        mar_value = mar(lms)

        pose = head_pose_deg(frame, lms)
        if pose is not None:
            pitch, yaw, roll = pose

        if not calibrated:
            if now - calib_start <= CALIB_SECONDS:
                if ear_value > 0:
                    calib_values.append(ear_value)
            else:
                if len(calib_values) >= 15:
                    baseline_ear = float(np.median(np.array(calib_values, dtype=np.float32)))
                    dyn_squint_min = baseline_ear * 0.70
                    dyn_squint_max = baseline_ear * 0.86
                    dyn_drowsy_thr = max(EAR_DROWSY_THRESHOLD, baseline_ear * 0.62)
                calibrated = True

        if ear_value < dyn_drowsy_thr:
            if closed_start is None:
                closed_start = now
            if now - closed_start >= EYE_CLOSED_SECONDS:
                drowsy_until = max(drowsy_until, now + DROWSY_HOLD_SECONDS)
        else:
            closed_start = None

        if dyn_squint_min <= ear_value <= dyn_squint_max:
            if squint_start is None:
                squint_start = now
            if now - squint_start >= SQUINT_SECONDS:
                squint_until = max(squint_until, now + SQUINT_HOLD_SECONDS)
        else:
            squint_start = None

        if mar_value > YAWN_THRESHOLD:
            if yawn_start is None:
                yawn_start = now
            if (now - yawn_start >= YAWN_MIN_SECONDS) and (now - last_yawn_time >= YAWN_COOLDOWN_SECONDS):
                last_yawn_time = now
                yawn_until = now + YAWN_HOLD_SECONDS
                yawn_events.append(now)
        else:
            yawn_start = None

        yawn_events = [t for t in yawn_events if now - t <= YAWN_WINDOW_SECONDS]
        yawns_5m = len(yawn_events)

        if yawns_5m >= YAWN_COUNT_TRIGGER and (now - last_yawn_drowsy_trigger >= YAWN_DROWSY_COOLDOWN_SECONDS):
            last_yawn_drowsy_trigger = now
            drowsy_until = max(drowsy_until, now + DROWSY_HOLD_SECONDS)

        pose_bad = False
        if pitch is not None and yaw is not None:
            if pitch > HEAD_DOWN_PITCH_DEG or abs(yaw) > LOOK_AWAY_YAW_DEG:
                pose_bad = True

        if pose_bad:
            if pose_start is None:
                pose_start = now
            if now - pose_start >= POSE_SECONDS:
                pose_until = max(pose_until, now + POSE_HOLD_SECONDS)
                drowsy_until = max(drowsy_until, now + 1.0)
        else:
            pose_start = None

        score = 100

        if closed_start is not None:
            score -= 40
        if now < drowsy_until:
            score -= 30
        if now < yawn_until:
            score -= 15
        if now < squint_until:
            score -= 10
        if now < pose_until:
            score -= 15

        if baseline_ear is not None:
            start = baseline_ear * 0.92
            end = baseline_ear * 0.70
            if ear_value < start:
                if ear_value <= end:
                    score -= 20
                else:
                    x = (start - ear_value) / (start - end)
                    score -= int(20 * x)

        if yawns_5m >= 2:
            score -= 5
        if yawns_5m >= 3:
            score -= 10

        if score < 0:
            score = 0

        draw_right(frame, f"FPS: {fps:.1f}", 50)
        draw_right(frame, f"EAR: {ear_value:.3f}", 90)
        draw_right(frame, f"MAR: {mar_value:.3f}", 130)
        draw_right(frame, f"Yawns(5m): {yawns_5m}", 170)
        if pitch is not None and yaw is not None:
            draw_right(frame, f"Pitch: {pitch:.1f}", 210)
            draw_right(frame, f"Yaw: {yaw:.1f}", 250)
        draw_right(frame, f"Score: {score}", 290)

        if not calibrated:
            left = max(0.0, CALIB_SECONDS - (now - calib_start))
            draw_right(frame, f"Calib: {left:.1f}s", 330)
        else:
            if muted:
                draw_right(frame, "Alarm: MUTED (m)", 330)
            else:
                draw_right(frame, "Alarm: ON (m)", 330)

        if closed_start is not None and now >= drowsy_until:
            draw_right(frame, f"Closed: {now - closed_start:.1f}s", 370)

    is_drowsy = now < drowsy_until

    if is_drowsy and not was_drowsy:
        pass
    was_drowsy = is_drowsy

    alarm_active = is_drowsy

    if alarm_active and (not muted):
        if now - last_beep_time >= BEEP_INTERVAL_SECONDS:
            last_beep_time = now
            beep()

    if now < yawn_until:
        cv2.putText(frame, "YAWN!", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)

    if now < squint_until:
        cv2.putText(frame, "SQUINT!", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 5)

    if now < pose_until:
        cv2.putText(frame, "INATTENTIVE!", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 0), 5)

    if now < drowsy_until:
        cv2.putText(frame, "DROWSY!", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)

    cv2.imshow("Driver Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        muted = not muted
        if not muted:
            last_beep_time = 0.0

cap.release()
cv2.destroyAllWindows()