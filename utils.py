import cv2, os
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
from PIL import Image
import io
import re
import base64

img_size = 150
model = load_model('models/model-001.model')
label_dict = {
    0: 'Peace Related',
    1: 'War Related'
}
def img_preprocessing(img):
    img = np.array(img)
    if(img.ndim == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray/255.0
    resized = cv2.resize(gray, (img_size, img_size))
    reshaped = resized.reshape(1, img_size, img_size)
    return reshaped

def predict(image):

    predict_image = img_preprocessing(image)
    prediction = model.predict(predict_image)
    
    result = np.argmax(prediction, axis = 1)[0]
    accuracy = float(np.max(prediction, axis = 1)[0])
    print(accuracy)
    label = label_dict[result]
    response = {
        'Result': {
            'Prediction': label,
            'Accuracy': accuracy
        }
    }
    return response
