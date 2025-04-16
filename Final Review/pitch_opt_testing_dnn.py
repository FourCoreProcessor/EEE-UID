import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from joblib import load
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger()

logger.info("Loading DNN model and scalers...")
model = tf.keras.models.load_model("trained_dnn_pitch_model.h5")

scaler_X = load("dnn_scaler_X.pkl")
scaler_y = load("dnn_scaler_y.pkl")
logger.info("Model and scalers loaded successfully.")

# GUI
class PitchAnglePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Wind Power Pitch Angle Predictor (DNN Version)")

        tk.Label(root, text="Wind Speed (m/s):").grid(row=0, column=0, padx=10, pady=5)
        self.wind_speed = tk.Entry(root)
        self.wind_speed.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(root, text="Wind Direction (°):").grid(row=1, column=0, padx=10, pady=5)
        self.wind_direction = tk.Entry(root)
        self.wind_direction.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(root, text="Angular Velocity (rad/s):").grid(row=2, column=0, padx=10, pady=5)
        self.angular_velocity = tk.Entry(root)
        self.angular_velocity.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(root, text="Flow Angle (Φ):").grid(row=3, column=0, padx=10, pady=5)
        self.flow_angle = tk.Entry(root)
        self.flow_angle.grid(row=3, column=1, padx=10, pady=5)

        tk.Button(root, text="Predict", command=self.predict_pitch_angle).grid(row=4, column=0, columnspan=2, pady=20)

    def predict_pitch_angle(self):
        try:
            wind_speed = float(self.wind_speed.get())
            wind_direction = float(self.wind_direction.get())
            angular_velocity = float(self.angular_velocity.get())
            flow_angle = float(self.flow_angle.get())

            input_data = np.array([[wind_speed, wind_direction, angular_velocity, flow_angle]])
            logger.info(f"Input data received: {input_data}")

            input_scaled = scaler_X.transform(input_data)
            logger.info(f"Scaled input: {input_scaled}")

            prediction_scaled = model.predict(input_scaled)
            logger.info(f"Scaled prediction: {prediction_scaled}")

            prediction = scaler_y.inverse_transform(prediction_scaled)
            logger.info(f"Final pitch angle prediction: {prediction[0][0]:.2f} degrees")

            messagebox.showinfo("Prediction Result", f"Predicted Optimum Pitch Angle: {prediction[0][0]:.2f} degrees")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Run app
if __name__ == "__main__":
    root = tk.Tk()
    app = PitchAnglePredictor(root)
    root.mainloop()
