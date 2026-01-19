import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATASET_PATH = "C:\Rishika\MajorProject_1\keypoints"
SEQUENCE_LENGTH = 30

actions = sorted(os.listdir(DATASET_PATH))
label_map = {label: idx for idx, label in enumerate(actions)}

X, y = [], []

for action in actions:
    action_path = os.path.join(DATASET_PATH, action)
    for seq in os.listdir(action_path):
        sequence = np.load(os.path.join(action_path, seq))
        if sequence.shape[0] == SEQUENCE_LENGTH:
            X.append(sequence)
            y.append(label_map[action])

X = np.array(X)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

#cnn lstm model:
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu',
           input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),

    LSTM(128, return_sequences=False),
    Dropout(0.4),

    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

# training it:
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

model.save("isl_keypoint_model_2.h5")
