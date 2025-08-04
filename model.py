import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# === Load CSV data ===
df = pd.read_csv("data.csv")  # Must have columns: Left, Right, Label

# === Features (X) and Target (y) ===
X = df[["Left", "Right"]].values
y = df["Label"].values  # Label should be -1, 0, or 1

# === Preprocess features ===
scaler_x = MinMaxScaler()
X = scaler_x.fit_transform(X)

# === Encode labels (from -1, 0, 1) to 0, 1, 2 ===
label_mapping = {-1: 0, 0: 1, 1: 2}
y = np.array([label_mapping[val] for val in y])
y = to_categorical(y, num_classes=3)  # One-hot encoding

# === Reshape X for Conv1D ===
X = X.reshape((X.shape[0], X.shape[1], 1))  # Shape: (samples, timesteps, channels)

# === Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Build 1D CNN Model ===
model = Sequential([
    Conv1D(16, kernel_size=1, activation='relu', input_shape=(2, 1)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Left, Straight, Right
])

# === Compile Model ===
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# === Train Model ===
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# === Save the model ===
model.save("direction_classifier_cnn.h5")

# === Predict example ===
predicted = model.predict(X_test[:1])
predicted_label = np.argmax(predicted)
reverse_mapping = {0: -1, 1: 0, 2: 1}
print("🧭 Predicted Direction Label:", reverse_mapping[predicted_label])
