from camera import Camera

import os
import cv2
import logging
from time import sleep
import numpy as np
import tensorflow as tf
from threading import Thread
import tensorflow.keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class MaskRecognition:

    def __init__(self):
        self.finalSize = 128
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.9
        self.thickness = 2
        self.lineType = 1
        self.probability = 100

        self.modelFace = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.modelMask =  keras.models.load_model("./models/bestModel.h5")

        self.camera = Camera()

    def faceRecognition(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frontalFaces = self.modelFace.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(128, 128),
                    flags = cv2.CASCADE_SCALE_IMAGE
        )
        return frontalFaces
    
    def maskRecognition(self, head):
        """ Obtaining prediction from CNN model """
        head = tf.image.resize_with_pad(head, self.finalSize, self.finalSize)
        head = tf.expand_dims(head, axis = 0)
        head = head / 255
        y_prob = np.around(self.modelMask.predict(head)[0] * 100, 1)
        self.probability = y_prob[0]

    def drawFrames(self, frame):
        frontalFaces = self.faceRecognition(frame)
        try:
            for (x, y, w, h) in frontalFaces:
                head = frame[y:y + h, x:x + w, :]
                Thread(target=self.maskRecognition(head)).start()
                probability = self.probability
                fontColor = (0, 255, 0)
                condition = "Mask"
                if probability < 50:
                    probability = 100 - probability
                    fontColor = (0, 0, 255)
                    condition = "No Mask"
                text = "%s %s" % (condition, np.around(probability, 1))
                cv2.rectangle(frame, (x, y), (x+w, y+h), fontColor, 5)
                bottomLeftCornerOfText = (x, y - 10)
                cv2.putText(frame, text,
                    bottomLeftCornerOfText,
                    self.font,
                    self.fontScale,
                    fontColor,
                    self.thickness,
                    self.lineType
                )
            cv2.imshow("Wear the Mask!", frame)
        except:
            cv2.imshow("Wear the Mask!", frame)

    def run(self):
        while True:
            frame = self.camera.getFrame()
            Thread(target=self.drawFrames(frame)).start()
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    def stop(self):
        self.camera.video.release()
        cv2.destroyAllWindows
        exit()

if __name__ == "__main__":
    recognition = MaskRecognition()
    recognition.run()
    recognition.stop()
