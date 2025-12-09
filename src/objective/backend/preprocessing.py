import numpy as np
from tensorflow.keras.preprocessing import image

IMG_HEIGHT = 150
IMG_WIDTH = 150

def preprocess_image(img_path):
    """
    Loads and preprocesses an image for prediction.
    """
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array