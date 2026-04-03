import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATASET_PATH = "dataset"
OUTPUT_PATH = "data"
MAX_FRAMES = 30

os.makedirs(OUTPUT_PATH, exist_ok=True)

BaseOptions = python.BaseOptions

hand = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        num_hands=2
    )
)

def extract_keypoints(hand_res):
    left = np.zeros(21*3)
    right = np.zeros(21*3)

    if hand_res.hand_landmarks:
        for i, lm in enumerate(hand_res.hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
            label = hand_res.handedness[i][0].category_name

            if label == "Left":
                left = pts
            else:
                right = pts

    return np.concatenate([left, right])


def normalize(seq):
    # centrer sur première main détectée
    for i in range(len(seq)):
        if np.any(seq[i] != 0):
            cx, cy = seq[i][0], seq[i][1]
            seq[i][::3] -= cx
            seq[i][1::3] -= cy
    return seq


def pad(seq):
    if len(seq) > MAX_FRAMES:
        return seq[:MAX_FRAMES]
    pad_len = MAX_FRAMES - len(seq)
    return np.vstack([seq, np.zeros((pad_len, seq.shape[1]))])


labels = os.listdir(DATASET_PATH)

for label in labels:
    os.makedirs(f"{OUTPUT_PATH}/{label}", exist_ok=True)

    for vid in os.listdir(f"{DATASET_PATH}/{label}"):
        path = f"{DATASET_PATH}/{label}/{vid}"
        cap = cv2.VideoCapture(path)
        seq = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            h_res = hand.detect(mp_img)
            kp = extract_keypoints(h_res)

            seq.append(kp)

        cap.release()

        seq = np.array(seq)
        seq = normalize(seq)
        seq = pad(seq)

        np.save(f"{OUTPUT_PATH}/{label}/{vid}.npy", seq)

print("✅ Extraction terminée")