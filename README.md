# Driver Drowsiness Detection System

Real-time multi-factor driver monitoring system built with Python, OpenCV and MediaPipe.

## Overview

This project detects driver drowsiness using:

- Eye Aspect Ratio (EAR)
- Yawn detection (MAR)
- Head Pose Estimation (PnP)
- Dynamic calibration
- Attention Score system
- Continuous alarm system
- Event logging
- CLI configuration
- Unit testing

The system analyzes facial landmarks in real-time and generates alerts when fatigue indicators are detected.

---

## Features

• Real-time face landmark detection  
• Eye closure detection  
• Squint detection  
• Yawn detection with time-window logic  
• Head pose estimation (pitch & yaw)  
• Dynamic calibration at startup  
• Continuous alarm (toggle with keyboard)  
• Attention Score (0–100)  
• Session event logging (CSV)  
• Command line interface support  
• Unit tests included  

---

## Architecture
src/
 main.py
 app/
 config.py
 geometry.py
 detectors.py
 pose.py
 alarm.py
 logger.py
└── run.py
config/
└── config.yaml
tests/
└── test_geometry.py

---

## Installation

Create virtual environment:
python3 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt
---

## Run

Basic run:
python src/main.py

Select camera:
python src/main.py –camera 0

Custom beep interval:
python src/main.py –beep-interval 0.5

Custom calibration time:
python src/main.py –calib-seconds 6
---

## Controls

- q → Quit
- m → Mute / Unmute alarm
- r → Recalibrate

---

## Logging

Each session creates:
logs/session_YYYYMMDD_HHMMSS.csv

Example log:
2025-05-19 14:33:10,yawn,0.61
2025-05-19 14:33:14,drowsy,
2025-05-19 14:33:15,beep,
---

## Attention Score

Score range: 0 – 100

Score decreases based on:

- Eye closure duration
- Yawning frequency
- Squint duration
- Head pose deviation
- Multi-event accumulation

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- PyYAML
- PyTest

---

## Future Improvements

- Audio-based fatigue detection
- Model optimization
- Embedded deployment (Jetson / Raspberry Pi)
- GUI version

---

Author: Enes Celik