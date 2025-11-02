"""
Real-time facial expression recognition using webcam
"""
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import argparse
import os
import time
from deepface import DeepFace
from utils import (download_haar_cascade, preprocess_image,
                  predict_emotion, draw_emotion_label, EMOTION_LABELS)


class RealTimeEmotionDetector:
    def __init__(self, model_path=None, use_pretrained=True):
        self.use_pretrained = use_pretrained
        self.model_path = model_path
        self.model = None
        self.face_cascade = None
        self.emotion_labels = list(EMOTION_LABELS.values())

        self._initialize_components()

    def _initialize_components(self):
        """Initialize model and face cascade"""
        # Initialize face cascade
        cascade_path = download_haar_cascade()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Load model
        if self.use_pretrained:
            print("Using DeepFace pre-trained model...")
        else:
            if self.model_path and os.path.exists(self.model_path):
                print(f"Loading custom model from {self.model_path}")
                self.model = load_model(self.model_path)
            else:
                print("Custom model not found, switching to DeepFace...")
                self.use_pretrained = True

    def detect_emotion_deepface(self, face_image):
        """Detect emotion using DeepFace"""
        try:
            result = DeepFace.analyze(
                face_image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                result = result[0]
            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion] / 100.0
            return emotion.capitalize(), confidence
        except Exception:
            return "Unknown", 0.0

    def detect_emotion_custom(self, face_image):
        """Detect emotion using custom trained model"""
        try:
            processed_face = preprocess_image(face_image)
            predictions = self.model.predict(processed_face, verbose=0)
            emotion_idx = np.argmax(predictions)
            confidence = predictions[0][emotion_idx]
            emotion_label = EMOTION_LABELS[emotion_idx]
            return emotion_label, confidence
        except Exception:
            return "Unknown", 0.0

    def process_frame(self, frame):
        """Process a single frame for emotion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            if self.use_pretrained:
                emotion, confidence = self.detect_emotion_deepface(face_roi)
            else:
                emotion, confidence = self.detect_emotion_custom(face_roi)

            frame = self.draw_results(frame, x, y, w, h, emotion, confidence)

        return frame

    def draw_results(self, frame, x, y, w, h, emotion, confidence):
        """Draw bounding box and emotion label on frame"""
        if confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        label = f"{emotion}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        cv2.rectangle(frame, (x, y-label_size[1]-10),
                      (x+label_size[0], y), color, -1)

        cv2.putText(frame, label, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def run(self, camera_index=0, display_fps=True):
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Real-time emotion detection started...")
        print("Press 'q' to quit, 's' to save screenshot")

        frame_count = 0
        fps_counter = cv2.getTickCount()
        fps = 0.0  # <-- initialize fps

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)

            if display_fps:
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / \
                        ((cv2.getTickCount() - fps_counter) /
                         cv2.getTickFrequency())
                    fps_counter = cv2.getTickCount()

                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(processed_frame, "Press 'q' to quit, 's' to save",
                        (10, processed_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Real-time Emotion Detection', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_name, processed_frame)
                print(f"Screenshot saved as {screenshot_name}")

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Facial Expression Recognition')
    parser.add_argument('--model', default='models/emotion_model.h5',
                        help='Path to trained model')
    parser.add_argument('--mode', choices=['pretrained', 'custom'], default='pretrained',
                        help='Use pretrained DeepFace model or custom trained model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--no-fps', action='store_true',
                        help='Hide FPS counter')

    args = parser.parse_args()

    use_pretrained = (args.mode == 'pretrained')
    detector = RealTimeEmotionDetector(
        model_path=args.model,
        use_pretrained=use_pretrained
    )

    detector.run(
        camera_index=args.camera,
        display_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()
