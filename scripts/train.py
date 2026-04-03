import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

DATA_PATH = "data"

labels = sorted(os.listdir(DATA_PATH))  # IMPORTANT
label_map = {label: i for i, label in enumerate(labels)}

X, y = [], []

for label in labels:
    for file in os.listdir(f"{DATA_PATH}/{label}"):
        seq = np.load(f"{DATA_PATH}/{label}/{file}")
        X.append(seq)
        y.append(label_map[label])

X = np.array(X)
y = to_categorical(y)

print("Dataset:", X.shape)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 126)),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=30, batch_size=8)

model.save("models/model.h5")

print("Labels:", labels)
print("✅ Modèle entraîné")