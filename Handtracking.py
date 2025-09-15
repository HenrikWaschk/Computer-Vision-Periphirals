import mediapipe as mp
import numpy as np
import cv2
from config import Handcount, Alpha,Model_complexity,EMA_Smoothing  # Alpha is smoothing factor (0 < Alpha <= 1)
from gestures import process_gestures, calculate_palmsize, calculate_average_index_vector, \
    control_mouse_with_right_hand, process_click_right_hand
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = mp.tasks.BaseOptions(model_asset_path="models/gesture_recognizer.task")
recognizer_options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE  # or VIDEO / LIVE_STREAM
)
recognizer = vision.GestureRecognizer.create_from_options(recognizer_options)


class HandTracker:
    HAND_COUNT = Handcount  # maximum number of hands to track
    def __init__(self, model_complexity=Model_complexity, detection_conf=0.5, tracking_conf=0.3):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.HAND_COUNT,
            model_complexity=model_complexity,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

        # EMA buffers for each hand
        self.ema_landmarks = {i: None for i in range(self.HAND_COUNT)}
        self.hand_size = 0.15

    # EMA smoothing
    def smooth_landmarks(self, current_landmarks, hand_index):
        current = np.array(current_landmarks)
        if self.ema_landmarks[hand_index] is None:
            self.ema_landmarks[hand_index] = current
        else:
            self.ema_landmarks[hand_index] = Alpha * current + (1 - Alpha) * self.ema_landmarks[hand_index]
        return self.ema_landmarks[hand_index]

    def process_frame(self, frame):
        """Process a BGR frame and return landmarks and annotated frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        gesture_result = recognizer.recognize(mp_image)

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            for hand_index in range(self.HAND_COUNT):
                if hand_index >= num_hands:
                    break
                hand_landmarks = results.multi_hand_landmarks[hand_index]
                handedness = results.multi_handedness[hand_index]

                hand_gestures = gesture_result.gestures[hand_index] if len(gesture_result.gestures) > hand_index else []

                label = handedness.classification[0].label
                score = handedness.classification[0].score
                if EMA_Smoothing:
                    # Convert landmarks to list
                    current_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    smoothed_landmarks = self.smooth_landmarks(current_landmarks, hand_index)

                    # Overwrite landmarks with smoothed values
                    for i, lm in enumerate(hand_landmarks.landmark):
                        lm.x, lm.y, lm.z = smoothed_landmarks[i]

                # Draw landmarks
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if hand_gestures:
                    top_gesture = hand_gestures[0].category_name
                    gesture_score = hand_gestures[0].score
                    #print(f"  - {hand_gestures[0].category_name} ({hand_gestures[0].score:.2f})")
                    if label == "Right" and top_gesture == "Open_Palm" and gesture_score > 0.5:
                        self.hand_size = calculate_palmsize(hand_landmarks)
                        #print("detecting palm")
                    else:
                        control_mouse_with_right_hand(hand_landmarks, handedness, self.hand_size)
                        process_click_right_hand(hand_landmarks, handedness, self.hand_size)

                # Display left/right label
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[0].x * w)
                y = int(hand_landmarks.landmark[0].y * h)

                index_vector = calculate_average_index_vector(hand_landmarks)
                index_vector[0] = index_vector[0] / self.hand_size
                index_vector[1] = index_vector[1] / self.hand_size * (9 / 16) + 0.1
                cv2.putText(frame, f'{label} ({score:.2f}) ({index_vector[0]:.2f}) ({index_vector[1]:.2f})', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        return results, frame



