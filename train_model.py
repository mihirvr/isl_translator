import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

DATA_DIR = 'dataset'
gestures = os.listdir(DATA_DIR)
X, y = [], []

for label in gestures:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        data = np.load(os.path.join(folder, file))
        X.append(data)
        y.append(label)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('gestures.txt', 'w') as f:
    for g in gestures:
        f.write(f"{g}\n")

print("✅ Model trained and saved as model.pkl")
