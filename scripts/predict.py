import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys
import os

# Load model and label encoder
model = load_model("models/fruit_model.h5")  
with open("models/label_encoder_v2.pkl", "rb") as f:  
    le = pickle.load(f)

def predict_image(img_path):
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    if not os.path.exists(img_path):
        print(f" File not found: {img_path}")
        return

    try:
        # Preprocess the image
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = le.inverse_transform([predicted_index])[0]
        confidence = prediction[0][predicted_index]

        # Output
        print(f"\n Predicted Spoilage Status: {predicted_class}")
        print(f" Confidence: {confidence:.2%}")
        print(f" Raw Prediction Index: {predicted_index}")
        print(f" LabelEncoder Classes: {list(le.classes_)}")

    except Exception as e:
        print(f"⚠️ Error during prediction: {e}")

# For quick CLI testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])