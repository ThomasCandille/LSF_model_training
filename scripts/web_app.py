import atexit
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

MODEL_PATH = "../models/model.h5"
LANDMARKER_PATH = "../models/hand_landmarker.task"
LABELS = ["bonjour", "non", "null"]
GAME_LABELS = ["bonjour", "non"]

MAX_FRAMES = 60
PREDICTION_HISTORY = 12
MIN_HISTORY_VALID = 8
MIN_VOTE_RATIO = 0.7
MIN_CONFIDENCE = 0.9
ANALYSIS_PAUSE_S = 1.2
CORRECT_COOLDOWN_S = 1.0


class SignGameEngine:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
                num_hands=2,
            )
        )

        self.cap = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        self.running = True

        self.sequence = []
        self.prediction_buffer = deque(maxlen=PREDICTION_HISTORY)
        self.analysis_resume_at = 0.0
        self.feedback_until = 0.0
        self.last_correct_time = 0.0

        self.latest_frame_jpeg = None
        self.state = {
            "target": np.random.choice(GAME_LABELS),
            "score": 0,
            "prediction": "Bougez vos mains pour commencer",
            "last_detected_label": "-",
            "last_detected_conf": 0.0,
            "feedback": "",
            "status": "Initialisation de la caméra",
            "history_ratio": 0.0,
            "analysis_pause_left": 0.0,
        }

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _extract_keypoints(self, hand_res):
        left = np.zeros(21 * 3)
        right = np.zeros(21 * 3)

        if hand_res.hand_landmarks:
            for i, lm in enumerate(hand_res.hand_landmarks):
                pts = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
                label = hand_res.handedness[i][0].category_name
                if label == "Left":
                    left = pts
                else:
                    right = pts

        return np.concatenate([left, right])

    def _normalize(self, seq):
        for i in range(len(seq)):
            if np.any(seq[i] != 0):
                cx, cy = seq[i][0], seq[i][1]
                seq[i][::3] -= cx
                seq[i][1::3] -= cy
        return seq

    def _pick_next_target(self, current_target):
        candidates = [label for label in GAME_LABELS if label != current_target]
        if candidates:
            return str(np.random.choice(candidates))
        return str(np.random.choice(GAME_LABELS))

    def _run_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            h_res = self.hand.detect(mp_img)

            kp = self._extract_keypoints(h_res)
            self.sequence.append(kp)
            self.sequence = self.sequence[-MAX_FRAMES:]

            now = time.time()
            with self.lock:
                target_label = self.state["target"]

            if len(self.sequence) == MAX_FRAMES and now >= self.analysis_resume_at:
                seq = np.array(self.sequence)
                seq = self._normalize(seq)
                seq = np.expand_dims(seq, axis=0)

                res = self.model.predict(seq, verbose=0)[0]
                pred = int(np.argmax(res))
                conf = float(res[pred])

                if conf > MIN_CONFIDENCE:
                    pred_label = LABELS[pred]
                    if pred_label in GAME_LABELS:
                        self.prediction_buffer.append((pred_label, conf))
                    else:
                        self.prediction_buffer.append((None, 0.0))
                else:
                    self.prediction_buffer.append((None, 0.0))

                valid_predictions = [item for item in self.prediction_buffer if item[0] is not None]
                with self.lock:
                    self.state["history_ratio"] = min(
                        1.0, len(valid_predictions) / max(1, MIN_HISTORY_VALID)
                    )

                if len(valid_predictions) >= MIN_HISTORY_VALID:
                    votes = {label: 0 for label in GAME_LABELS}
                    for label, _ in valid_predictions:
                        votes[label] += 1

                    stable_label = max(votes, key=votes.get)
                    vote_ratio = votes[stable_label] / len(valid_predictions)

                    if vote_ratio >= MIN_VOTE_RATIO:
                        stable_conf = float(
                            np.mean(
                                [
                                    conf_v
                                    for label, conf_v in valid_predictions
                                    if label == stable_label
                                ]
                            )
                        )
                        with self.lock:
                            self.state["prediction"] = f"{stable_label} ({stable_conf:.2f})"
                            self.state["last_detected_label"] = stable_label
                            self.state["last_detected_conf"] = stable_conf

                        self.analysis_resume_at = now + ANALYSIS_PAUSE_S
                        self.prediction_buffer.clear()
                        self.sequence.clear()

                        with self.lock:
                            if stable_label == target_label:
                                if now - self.last_correct_time > CORRECT_COOLDOWN_S:
                                    self.state["score"] += 1
                                    self.state["feedback"] = "Réussi"
                                    self.state["status"] = "Mouvement parfait"
                                    self.feedback_until = now + 1.0
                                    self.last_correct_time = now
                                    self.state["target"] = self._pick_next_target(target_label)
                            else:
                                self.state["feedback"] = "Essayez encore"
                                self.state["status"] = "Bon mouvement, mauvais signe"
                                self.feedback_until = now + 0.8
                    else:
                        with self.lock:
                            self.state["prediction"] = "Analyse du mouvement..."
                            self.state["status"] = "Collecte d'images cohérentes"
                else:
                    with self.lock:
                        self.state["prediction"] = "Analyse du mouvement..."
                        self.state["status"] = "Collecte du mouvement"
            elif now < self.analysis_resume_at:
                wait_left = max(0.0, self.analysis_resume_at - now)
                with self.lock:
                    self.state["prediction"] = f"Nouvelle analyse dans {wait_left:.1f}s"
                    self.state["status"] = "Attendez la réinitialisation avant le prochain geste"
                    self.state["analysis_pause_left"] = wait_left

            if now > self.feedback_until:
                with self.lock:
                    self.state["feedback"] = ""

            with self.lock:
                if now >= self.analysis_resume_at:
                    self.state["analysis_pause_left"] = 0.0
                target_text = self.state["target"].upper()
                score_text = str(self.state["score"])

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (17, 20, 33), -1)
            cv2.putText(frame, f"Cible : {target_text}", (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (77, 212, 255), 2)
            cv2.putText(frame, f"Score : {score_text}", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (92, 255, 173), 2)

            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with self.lock:
                    self.latest_frame_jpeg = buffer.tobytes()

            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.latest_frame_jpeg

    def get_state(self):
        with self.lock:
            return dict(self.state)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)
engine = SignGameEngine()


@atexit.register
def _cleanup():
    engine.stop()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = engine.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify(engine.get_state())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
