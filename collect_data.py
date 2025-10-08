import cv2
import os
import time
import mediapipe as mp
import numpy as np

GESTURES = ['hello', 'sorry', 'thankyou', 'yes', 'no']
SAVE_PATH = 'dataset'
FRAMES_PER_GESTURE = 30  # ~10 FPS for 3 seconds

os.makedirs(SAVE_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    os.makedirs(os.path.join(SAVE_PATH, gesture), exist_ok=True)
    input(f"\nPress ENTER to start capturing video for gesture: '{gesture}'")
    print("⚠️ Starting in 3 seconds. Get ready!")
    time.sleep(3)

    collected = 0
    while collected < FRAMES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                lm = handLms.landmark
                data = [pt.x for pt in lm] + [pt.y for pt in lm] + [pt.z for pt in lm]
                save_file = os.path.join(SAVE_PATH, gesture, f"{int(time.time()*1000)}.npy")
                np.save(save_file, np.array(data))
                collected += 1
                print(f"✅ Frame {collected} saved")
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Recording '{gesture}' - Frame {collected}/{FRAMES_PER_GESTURE}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Recording Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped manually.")
            break

print(" Data collection complete")
cap.release()
cv2.destroyAllWindows()
