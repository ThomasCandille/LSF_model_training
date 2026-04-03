import cv2
import numpy as np
import mediapipe as mp
import random
import time
from collections import deque
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/model.h5"
MAX_FRAMES = 60
PREDICTION_HISTORY = 12
MIN_HISTORY_VALID = 8
MIN_VOTE_RATIO = 0.7
MIN_CONFIDENCE = 0.9
ANALYSIS_PAUSE_S = 1.2
WINDOW_NAME = "LSF Sign Arena"
UI_WIDTH = 1280
UI_HEIGHT = 720
CAM_W = 800
CAM_H = 600
CAM_X = 40
CAM_Y = 70

labels = ["bonjour", "non", "null"]  # DOIT matcher train

model = load_model(MODEL_PATH)

BaseOptions = python.BaseOptions

hand = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        num_hands=2
    )
)


def draw_gradient_background(canvas, top_color, bottom_color):
    h, w = canvas.shape[:2]
    for y in range(h):
        t = y / max(1, h - 1)
        color = [
            int(top_color[c] * (1.0 - t) + bottom_color[c] * t)
            for c in range(3)
        ]
        cv2.line(canvas, (0, y), (w, y), color, 1)


def draw_card(canvas, x, y, w, h, color=(25, 25, 25), border=(90, 90, 90), alpha=0.9):
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 2)


def draw_progress_bar(canvas, x, y, w, h, ratio, fg=(0, 220, 255), bg=(40, 40, 40)):
    ratio = max(0.0, min(1.0, ratio))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + int(w * ratio), y + h), fg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (120, 120, 120), 1)


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
    for i in range(len(seq)):
        if np.any(seq[i] != 0):
            cx, cy = seq[i][0], seq[i][1]
            seq[i][::3] -= cx
            seq[i][1::3] -= cy
    return seq


sequence = []
cap = cv2.VideoCapture(0)
prediction = ""
score = 0
game_labels = ["bonjour", "non"]
target_label = random.choice(game_labels)
feedback = ""
feedback_color = (255, 255, 255)
feedback_until = 0.0
last_correct_time = 0.0
correct_cooldown_s = 1.0
prediction_buffer = deque(maxlen=PREDICTION_HISTORY)
analysis_resume_at = 0.0
last_detected_label = "-"
last_detected_conf = 0.0
status_text = "Move your hands to start"
status_color = (180, 180, 180)


def pick_next_target(current_target, choices):
    # Avoid repeating the same prompt twice in a row for a clearer game rhythm.
    candidates = [label for label in choices if label != current_target]
    if candidates:
        return random.choice(candidates)
    return random.choice(choices)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    h_res = hand.detect(mp_img)
    kp = extract_keypoints(h_res)

    sequence.append(kp)
    sequence = sequence[-MAX_FRAMES:]

    now = time.time()
    if len(sequence) == MAX_FRAMES and now >= analysis_resume_at:
        seq = np.array(sequence)
        seq = normalize(seq)
        seq = np.expand_dims(seq, axis=0)

        res = model.predict(seq, verbose=0)[0]
        pred = np.argmax(res)
        conf = res[pred]
        if conf > MIN_CONFIDENCE:
            pred_label = labels[pred]
            if pred_label in game_labels:
                prediction_buffer.append((pred_label, conf))
            else:
                prediction_buffer.append((None, 0.0))
        else:
            prediction_buffer.append((None, 0.0))

        valid_predictions = [item for item in prediction_buffer if item[0] is not None]
        if len(valid_predictions) >= MIN_HISTORY_VALID:
            votes = {label: 0 for label in game_labels}
            for label, _ in valid_predictions:
                votes[label] += 1

            stable_label = max(votes, key=votes.get)
            vote_ratio = votes[stable_label] / len(valid_predictions)

            if vote_ratio >= MIN_VOTE_RATIO:
                stable_conf = np.mean([conf_v for label, conf_v in valid_predictions if label == stable_label])
                prediction = f"{stable_label} ({stable_conf:.2f})"
                last_detected_label = stable_label
                last_detected_conf = float(stable_conf)
                analysis_resume_at = now + ANALYSIS_PAUSE_S
                prediction_buffer.clear()
                sequence.clear()

                if stable_label == target_label:
                    if now - last_correct_time > correct_cooldown_s:
                        score += 1
                        feedback = "Correct!"
                        feedback_color = (0, 255, 0)
                        feedback_until = now + 1.0
                        last_correct_time = now
                        target_label = pick_next_target(target_label, game_labels)
                        status_text = "Perfect movement"
                        status_color = (0, 255, 0)
                else:
                    feedback = "Try again"
                    feedback_color = (0, 0, 255)
                    feedback_until = now + 0.7
                    status_text = "Good motion, wrong sign"
                    status_color = (0, 100, 255)
            else:
                prediction = "Analyzing movement..."
                status_text = "Collecting consistent frames"
                status_color = (0, 220, 255)
        else:
            prediction = "Analyzing movement..."
            status_text = "Collecting movement"
            status_color = (0, 220, 255)
    elif now < analysis_resume_at:
        wait_left = max(0.0, analysis_resume_at - now)
        prediction = f"Next analysis in {wait_left:.1f}s"
        status_text = "Hold and reset for next motion"
        status_color = (255, 200, 0)

    if now > feedback_until:
        feedback = ""

    canvas = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
    draw_gradient_background(canvas, (22, 26, 44), (8, 10, 20))

    cam_frame = cv2.resize(frame, (CAM_W, CAM_H))
    draw_card(canvas, CAM_X - 10, CAM_Y - 10, CAM_W + 20, CAM_H + 20,
              color=(18, 18, 28), border=(70, 120, 180), alpha=0.95)
    canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam_frame
    cv2.putText(canvas, "Live Camera", (CAM_X, CAM_Y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 255), 2)

    panel_x = CAM_X + CAM_W + 40
    panel_y = CAM_Y
    panel_w = 360

    draw_card(canvas, panel_x, panel_y, panel_w, 130, color=(26, 30, 48), border=(90, 150, 220), alpha=0.95)
    cv2.putText(canvas, "Target Sign", (panel_x + 16, panel_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 190, 255), 2)
    cv2.putText(canvas, target_label.upper(), (panel_x + 16, panel_y + 92),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 220, 255), 3)

    draw_card(canvas, panel_x, panel_y + 150, panel_w, 110, color=(24, 36, 30), border=(90, 180, 130), alpha=0.95)
    cv2.putText(canvas, "Score", (panel_x + 16, panel_y + 184),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 230, 200), 2)
    cv2.putText(canvas, str(score), (panel_x + 16, panel_y + 242),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, (80, 255, 170), 3)

    draw_card(canvas, panel_x, panel_y + 280, panel_w, 140, color=(34, 26, 32), border=(190, 110, 170), alpha=0.95)
    cv2.putText(canvas, "Detected", (panel_x + 16, panel_y + 312),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 180, 220), 2)
    cv2.putText(canvas, f"{last_detected_label} ({last_detected_conf:.2f})", (panel_x + 16, panel_y + 348),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 230, 245), 2)
    cv2.putText(canvas, prediction, (panel_x + 16, panel_y + 392),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 220, 220), 2)

    draw_card(canvas, panel_x, panel_y + 440, panel_w, 130, color=(20, 20, 24), border=(120, 120, 120), alpha=0.95)
    cv2.putText(canvas, "Movement Confidence", (panel_x + 16, panel_y + 472),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 220, 220), 2)
    history_ratio = len([item for item in prediction_buffer if item[0] is not None]) / max(1, MIN_HISTORY_VALID)
    draw_progress_bar(canvas, panel_x + 16, panel_y + 492, panel_w - 32, 20, history_ratio, fg=(0, 220, 255))

    if now < analysis_resume_at:
        wait_ratio = 1.0 - ((analysis_resume_at - now) / ANALYSIS_PAUSE_S)
        draw_progress_bar(canvas, panel_x + 16, panel_y + 524, panel_w - 32, 16, wait_ratio, fg=(255, 200, 0))
        cv2.putText(canvas, "Reset timer", (panel_x + 16, panel_y + 560),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 220, 120), 2)

    cv2.putText(canvas, status_text, (CAM_X, UI_HEIGHT - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(canvas, "Press Q to quit", (UI_WIDTH - 220, UI_HEIGHT - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    if feedback:
        pulse = 1.0 + 0.1 * np.sin(now * 10.0)
        thickness = 2 if pulse < 1.0 else 3
        cv2.putText(canvas, feedback, (panel_x + 16, panel_y + 620),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, feedback_color, thickness)

    cv2.imshow(WINDOW_NAME, canvas)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()