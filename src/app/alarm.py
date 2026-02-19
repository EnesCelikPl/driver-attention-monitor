import subprocess

def beep():
    try:
        subprocess.Popen(["osascript", "-e", "beep 1"])
    except Exception:
        pass