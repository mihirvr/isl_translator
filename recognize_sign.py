import cv2
import mediapipe as mp
import numpy as np
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("gestures.txt", "r") as f:
    labels = [line.strip() for line in f]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

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
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark
            data = [pt.x for pt in lm] + [pt.y for pt in lm] + [pt.z for pt in lm]

            if len(data) == 63:
                pred = model.predict([data])[0]
                cv2.putText(frame, f"Detected: {pred}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    cv2.imshow("ISL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
