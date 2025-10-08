import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
from tkinter import *
from PIL import Image, ImageTk
from collections import deque

# Load model and labels
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("gestures.txt", "r") as f:
    labels = [line.strip() for line in f]


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()

# GUI setup
root = Tk()
root.title("ISL Translator (5 Gestures)")
root.geometry("700x550")

video_label = Label(root)
video_label.pack()
text_label = Label(root, text="", font=("Arial", 24), pady=10)
text_label.pack()

cap = cv2.VideoCapture(0)

# Voice debounce
prediction_history = deque(maxlen=5)
last_spoken = ""

def update_frame():
    global last_spoken
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = ""

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark
            data = [pt.x for pt in lm] + [pt.y for pt in lm] + [pt.z for pt in lm]

            if len(data) == 63:
                pred = model.predict([data])[0]
                prediction = pred
                prediction_history.append(pred)


    if prediction_history:
        most_common = max(set(prediction_history), key=prediction_history.count)
        count = prediction_history.count(most_common)
        if count >= 3 and most_common != last_spoken:
            text_label.config(text=f"Detected: {most_common}")
            engine.say(most_common)
            engine.runAndWait()
            last_spoken = most_common
    else:
        text_label.config(text="")


    img = Image.fromarray(rgb)
    img = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img
    video_label.configure(image=img)

    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
