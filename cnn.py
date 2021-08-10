from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

imageSize = 150
cnn = load_model('models/model-001.model')
label = {
    'Peace': 0,
    'War': 1
}

def preprocessImg(img):
    img = np.array(img)
    if (img.ndim == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    gray = gray/255
    resized = cv2.resize(gray, (imageSize, imageSize))
    reshaped = resized.reshape(1, imageSize, imageSize)
    return reshaped
