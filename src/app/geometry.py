import numpy as np

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

def mar(landmarks, top_i, bot_i, left_i, right_i):
    top = np.array([landmarks[top_i].x, landmarks[top_i].y])
    bottom = np.array([landmarks[bot_i].x, landmarks[bot_i].y])
    left = np.array([landmarks[left_i].x, landmarks[left_i].y])
    right = np.array([landmarks[right_i].x, landmarks[right_i].y])
    v = dist(top, bottom)
    h = dist(left, right)
    if h == 0:
        return 0.0
    return v / h