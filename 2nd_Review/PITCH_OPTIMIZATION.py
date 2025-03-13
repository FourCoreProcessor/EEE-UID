import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv(r"C:\Users\jkyog\OneDrive\Desktop\college\S2\EEE UID\Wind_Speed_After using formula.csv")

X = df[['Wind Speed (m/s)', 'Wind Direction (\u00b0)', 'Angular_Velocity (rad/s)', 'Flow_Angle (\u03a6)']]
y = df[['Proxy_Blade_Angle']]

# Normalizing the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Building ANN Model with Dropout
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input Layer
    keras.layers.Dropout(0.3),  # Dropout Layer 1
    keras.layers.Dense(32, activation='relu'),  # Hidden Layer 1
    keras.layers.Dropout(0.3),  # Dropout Layer 2
    keras.layers.Dense(16, activation='relu'),  # Hidden Layer 2
    keras.layers.Dropout(0.3),  # Dropout Layer 3
    keras.layers.Dense(1, activation='linear')  # Output Layer (predicting PitchDeg)
])

# Compiling the model
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanAbsoluteError()])

# Training the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the model
model.save("trained_wind_speed_model.h5")
print("Model saved as 'trained_wind_speed_model.h5'")

loss, mae = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred)
r2 = r2_score(y_test_original, y_pred_original)

print(f"Test MAE: {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


from joblib import dump
dump(scaler_X, "scaler_X.pkl")
dump(scaler_y, "scaler_y.pkl")