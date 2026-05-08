import cv2
import numpy as np


def preprocess(image):

    image = cv2.resize(image, (128, 128))

    image = image.astype(np.float32)

    image = image / 255.0

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    return image