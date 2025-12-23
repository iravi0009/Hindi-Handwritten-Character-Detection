import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("../model/hindi_cnn_model.h5")
labels = np.load("../model/labels.npy")

def predict_character(char_img):
    img = cv2.resize(char_img, (64,64)) / 255.0
    img = img.reshape(1,64,64,1)
    pred = model.predict(img)
    return labels[np.argmax(pred)]
