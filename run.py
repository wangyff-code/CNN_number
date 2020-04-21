from tensorflow_core.python.keras.models import load_model

import cv2
import numpy as np
import tensorflow as tf



class my_CNN_class():
    def __init__(self):
        print("Using loaded model to predict...")
        self.load_model = load_model("iris_model.h5")

    def dect_number(self,img):
        img = cv2.resize(img,(28,28))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.bitwise_not(img)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = np.reshape(img,(1,28,28,1))
        predicted = self.load_model.predict(img)
        n = np.argmax(predicted[0])
        img = np.reshape(img,(28,28))
        if predicted[0][n] <0.2:
            n =None
        return n,img

if __name__ == "__main__":
    my_cnn = my_CNN_class()
    img = cv2.imread('my_img/3.jpg')
    n,img = my_cnn.dect_number(img)
    cv2.imshow('0',img)
    print(n)
    cv2.waitKey(0)