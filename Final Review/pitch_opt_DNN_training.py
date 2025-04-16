import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from joblib import dump
import matplotlib.pyplot as plt
import tf2onnx
import logging
import os
import shutil
import onnx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger()

# Load dataset
logger.info("Loading dataset...")
df = pd.read_csv("Wind_Speed_After using formula.csv")

# Split into features and target
X = df[['Wind Speed (m/s)', 'Wind Direction (°)', 'Angular_Velocity (rad/s)', 'Flow_Angle (Φ)']]
y = df[['Proxy_Blade_Angle']]

# Normalize inputs and target
logger.info("Normalizing input and output...")
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train/test split
logger.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build DNN model
logger.info("Building Deep Neural Network...")
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(X_train.shape[1],), name='input_layer'),
    keras.layers.Dense(256, activation='relu', name='dense_1'),
    keras.layers.Dropout(0.2, name='dropout_1'),
    keras.layers.Dense(128, activation='relu', name='dense_2'),
    keras.layers.Dropout(0.2, name='dropout_2'),
    keras.layers.Dense(64, activation='relu', name='dense_3'),
    keras.layers.Dropout(0.1, name='dropout_3'),
    keras.layers.Dense(32, activation='relu', name='dense_4'),
    keras.layers.Dense(1, activation='linear', name='output_layer')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse', metrics=['mae'])

# Train model
logger.info("Training model...")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=200, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# Save scalers
logger.info("Saving scalers...")
dump(scaler_X, "dnn_scaler_X.pkl")
dump(scaler_y, "dnn_scaler_y.pkl")
logger.info("Scalers saved successfully.")

# Save dropout-free inference model
logger.info("Creating and saving dropout-free inference model...")
inference_model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(X_train.shape[1],), name='input_layer'),
    keras.layers.Dense(256, activation='relu', name='dense_1'),
    keras.layers.Dense(128, activation='relu', name='dense_2'),
    keras.layers.Dense(64, activation='relu', name='dense_3'),
    keras.layers.Dense(32, activation='relu', name='dense_4'),
    keras.layers.Dense(1, activation='linear', name='output_layer')
])
inference_model.set_weights(model.get_weights())
inference_model.save("inference_dnn_pitch_model.h5")

# Save as TensorFlow SavedModel
logger.info("Saving model in TensorFlow SavedModel format for ONNX conversion...")
tf.saved_model.save(inference_model, "temp_saved_model")

# Convert to ONNX
logger.info("Converting to ONNX format...")
try:
    input_spec = (tf.TensorSpec([None, X_train.shape[1]], tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(
        inference_model,
        input_signature=input_spec,
        opset=13,
        output_path="pitch_model.onnx"
    )
    logger.info("Model successfully converted to ONNX format.")
except Exception as e:
    logger.error(f"ONNX conversion failed: {e}")

# Verify ONNX
try:
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load("pitch_model.onnx")
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model is valid ✅")
except Exception as e:
    logger.error(f"ONNX model verification failed ❌: {e}")

# Evaluate
logger.info("Evaluating model on test data...")
loss, mae = model.evaluate(X_test, y_test)
logger.info(f"Test MAE: {mae:.4f}")
y_pred = model.predict(X_test)
y_pred_orig = scaler_y.inverse_transform(y_pred)
y_test_orig = scaler_y.inverse_transform(y_test)
r2 = r2_score(y_test_orig, y_pred_orig)
logger.info(f"R² Score: {r2:.4f}")

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
