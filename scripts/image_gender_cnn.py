#
# Kyler Robison
#
# VM script for deployment of image model for gender.
#
#

from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np


class Model:
    input_dir = ""
    model = None
    face_cascade = None

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.model = load_model('image_gender_cnn.hdf5')
        self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    def classify(self, user_id):
        image = cv.imread(self.input_dir + '/image/' + user_id + '.jpg')
        faces = self.face_cascade.detectMultiScale(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 1.1, 4)

        # if not one face, go for baseline, later on defer to another model
        if len(faces) != 1:
            return "female"

        result_image = None
        for (x, y, w, h) in faces:
            result_image = cv.resize(image[y:y + h, x:x + w], (96, 96), interpolation=cv.INTER_CUBIC)

        # convert to grayscale
        result_image = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)
        result_image = cv.resize(result_image, (96, 96))

        # normalize pixel values to between 0 and 1
        result_image = result_image.astype('float32')
        result_image /= 255

        result_image = np.expand_dims(result_image, axis=0)

        result = self.model.predict(result_image, verbose=0)
        class_idx = np.argmax(result)

        if class_idx == 0:
            return "male"
        else:
            return "female"
