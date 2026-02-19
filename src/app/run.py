import cv2
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .detectors import ear_avg, mouth_ratio
from .pose import head_pose_deg
from .alarm import beep
from .logger import new_session_path, write_header, log_event

def run_app(cfg, args):
    thr = cfg["thresholds"]
    calib_cfg = cfg["calibration"]
    alarm_cfg = cfg["alarm"]

    camera_index = args.camera if args.camera is not None else int(cfg.get("camera_index", 0))
    beep_interval = args.beep_interval if args.beep_interval is not None else float(alarm_cfg["beep_interval_seconds"])
    calib_seconds = args.calib_seconds if args.calib_seconds is not None else float(calib_cfg["seconds"])

    model_path = args.model
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_index)

    EAR_DROWSY_BASE = float(thr["ear_drowsy_base"])
    EYE_CLOSED_SECONDS = float(thr["eye_closed_seconds"])
    DROWSY_HOLD_SECONDS = float(thr["drowsy_hold_seconds"])

    YAWN_THRESHOLD = float(thr["yawn_threshold"])
    YAWN_MIN_SECONDS = float(thr["yawn_min_seconds"])
    YAWN_HOLD_SECONDS = float(thr["yawn_hold_seconds"])
    YAWN_WINDOW_SECONDS = float(thr["yawn_window_seconds"])
    YAWN_COUNT_TRIGGER = int(thr["yawn_count_trigger"])
    YAWN_COOLDOWN_SECONDS = float(thr["yawn_cooldown_seconds"])
    YAWN_DROWSY_COOLDOWN_SECONDS = float(thr["yawn_drowsy_cooldown_seconds"])

    POSE_SECONDS = float(thr["pose_seconds"])
    POSE_HOLD_SECONDS = float(thr["pose_hold_seconds"])
    HEAD_DOWN_PITCH_DEG = float(thr["head_down_pitch_deg"])
    LOOK_AWAY_YAW_DEG = float(thr["look_away_yaw_deg"])

    SQUINT_SECONDS = float(thr["squint_seconds"])
    SQUINT_HOLD_SECONDS = float(thr["squint_hold_seconds"])

    closed_start = None
    yawn_start = None
    pose_start = None
    squint_start = None

    yawn_until = 0.0
    pose_until = 0.0
    squint_until = 0.0
    drowsy_until = 0.0

    last_yawn_time = 0.0
    last_yawn_drowsy_trigger = 0.0
    yawn_events = []

    muted = False
    last_beep_time = 0.0

    calib_start = time.time()
    calib_values = []
    calibrated = False
    baseline_ear = None
    dyn_drowsy_thr = EAR_DROWSY_BASE
    dyn_squint_min = None
    dyn_squint_max = None

    session = new_session_path()
    write_header(session)

    last_fps_time = time.time()
    fps = 0.0

    def draw_right(frame, text, y, scale=0.8, thickness=2):
        h, w = frame.shape[:2]
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = w - tw - 20
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)

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

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            ear_value = ear_avg(lms)
            mar_value = mouth_ratio(lms)

            pose = head_pose_deg(frame, lms)
            if pose is not None:
                pitch, yaw, _ = pose

            if (not calibrated) and (now - calib_start <= calib_seconds):
                if ear_value and ear_value > 0:
                    calib_values.append(ear_value)
            elif not calibrated:
                if len(calib_values) >= 15:
                    baseline_ear = float(np.median(np.array(calib_values, dtype=np.float32)))
                    dyn_drowsy_thr = max(EAR_DROWSY_BASE, baseline_ear * 0.62)
                    dyn_squint_min = baseline_ear * 0.70
                    dyn_squint_max = baseline_ear * 0.86
                calibrated = True
                log_event(session, "calibration_done", f"{baseline_ear if baseline_ear else ''}")

            if ear_value < dyn_drowsy_thr:
                if closed_start is None:
                    closed_start = now
                if now - closed_start >= EYE_CLOSED_SECONDS:
                    drowsy_until = max(drowsy_until, now + DROWSY_HOLD_SECONDS)
            else:
                closed_start = None

            if dyn_squint_min is not None and dyn_squint_max is not None:
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
                    log_event(session, "yawn", f"{mar_value:.3f}")
            else:
                yawn_start = None

            yawn_events = [t for t in yawn_events if now - t <= YAWN_WINDOW_SECONDS]
            yawns_5m = len(yawn_events)

            if yawns_5m >= YAWN_COUNT_TRIGGER and (now - last_yawn_drowsy_trigger >= YAWN_DROWSY_COOLDOWN_SECONDS):
                last_yawn_drowsy_trigger = now
                drowsy_until = max(drowsy_until, now + DROWSY_HOLD_SECONDS)
                log_event(session, "drowsy_from_yawns", str(yawns_5m))

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

            if args.log_score:
                log_event(session, "score", str(score))

            draw_right(frame, f"FPS: {fps:.1f}", 50)
            draw_right(frame, f"EAR: {ear_value:.3f}", 90)
            draw_right(frame, f"MAR: {mar_value:.3f}", 130)
            draw_right(frame, f"Yawns(5m): {yawns_5m}", 170)
            if pitch is not None and yaw is not None:
                draw_right(frame, f"Pitch: {pitch:.1f}", 210)
                draw_right(frame, f"Yaw: {yaw:.1f}", 250)
            draw_right(frame, f"Score: {score}", 290)

            if not calibrated:
                left = max(0.0, calib_seconds - (now - calib_start))
                draw_right(frame, f"Calib: {left:.1f}s", 330)
            else:
                draw_right(frame, "Calib: OK", 330)

            draw_right(frame, "m: mute  r: recalib  q: quit", 370, scale=0.6, thickness=1)

        is_drowsy = now < drowsy_until

        if is_drowsy and (not muted):
            if now - last_beep_time >= beep_interval:
                last_beep_time = now
                beep()
                log_event(session, "beep", "")

        if now < yawn_until:
            cv2.putText(frame, "YAWN!", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)
        if now < squint_until:
            cv2.putText(frame, "SQUINT!", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 5)
        if now < pose_until:
            cv2.putText(frame, "INATTENTIVE!", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 0), 5)
        if now < drowsy_until:
            cv2.putText(frame, "DROWSY!", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
            log_event(session, "drowsy", "")

        cv2.imshow("Driver Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            muted = not muted
            log_event(session, "mute", "on" if muted else "off")
            if not muted:
                last_beep_time = 0.0
        if key == ord("r"):
            calib_start = time.time()
            calib_values = []
            calibrated = False
            baseline_ear = None
            dyn_drowsy_thr = EAR_DROWSY_BASE
            dyn_squint_min = None
            dyn_squint_max = None
            log_event(session, "recalibration", "")

    cap.release()
    cv2.destroyAllWindows()