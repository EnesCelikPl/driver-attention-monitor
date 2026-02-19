import numpy as np
import cv2
import math

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