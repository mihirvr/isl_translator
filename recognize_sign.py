import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load gesture labels
with open("gestures.txt", "r") as f:
    labels = [line.strip() for line in f]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract all x, y, z coordinates
            lm = handLms.landmark
            data = [pt.x for pt in lm] + [pt.y for pt in lm] + [pt.z for pt in lm]

            # Calculate bounding box coordinates
            x_coords = [int(pt.x * w) for pt in lm]
            y_coords = [int(pt.y * h) for pt in lm]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Predict gesture
            if len(data) == 63:
                pred = model.predict([data])[0]

                # Draw box around the hand
                cv2.rectangle(frame, (x_min - 20, y_min - 20),
                              (x_max + 20, y_max + 20), (0, 255, 0), 2)

                # Add text below the box
                text_y = y_max + 40 if y_max + 40 < h else h - 10
                cv2.putText(frame, f"Detected: {pred}",
                            (x_min, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show result
    cv2.imshow("ISL Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
