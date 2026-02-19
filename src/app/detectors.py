from .geometry import ear_one, mar

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

def ear_avg(landmarks):
    return (ear_one(landmarks, LEFT_EYE) + ear_one(landmarks, RIGHT_EYE)) / 2.0

def mouth_ratio(landmarks):
    return mar(landmarks, MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT)