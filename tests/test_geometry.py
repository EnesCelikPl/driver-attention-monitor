import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from types import SimpleNamespace
from src.app.geometry import ear_one, mar

def lm(x, y):
    return SimpleNamespace(x=x, y=y)

def test_ear_one_positive():
    l = [lm(0, 0) for _ in range(500)]
    idx = [0, 1, 2, 3, 4, 5]
    l[0] = lm(0.1, 0.5)
    l[3] = lm(0.9, 0.5)
    l[1] = lm(0.3, 0.4)
    l[5] = lm(0.3, 0.6)
    l[2] = lm(0.7, 0.4)
    l[4] = lm(0.7, 0.6)
    v = ear_one(l, idx)
    assert v > 0.0

def test_mar_positive():
    l = [lm(0, 0) for _ in range(500)]
    top, bot, left, right = 10, 11, 12, 13
    l[top] = lm(0.5, 0.4)
    l[bot] = lm(0.5, 0.6)
    l[left] = lm(0.3, 0.5)
    l[right] = lm(0.7, 0.5)
    v = mar(l, top, bot, left, right)
    assert v > 0.0