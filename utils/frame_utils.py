# deepfake_platform/utils/frame_utils.py
import cv2
import os

def sample_frames_from_video(path, max_frames=16):
    """
    Returns: list of BGR frames (numpy arrays)
    """
    frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # fallback: just read up to max_frames sequentially
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    else:
        step = max(1, total // max_frames)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
                if len(frames) >= max_frames:
                    break
            idx += 1
    cap.release()
    return frames

def save_bytes_to_tempfile(file_bytes, filename="temp_video.mp4"):
    path = os.path.join(os.getcwd(), filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path
