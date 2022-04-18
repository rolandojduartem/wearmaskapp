import cv2

class Camera:

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def getFrame(self):
        _, frame = self.video.read()
        frame = cv2.flip(frame, 1)
        return frame

    def showFrame(self, frame):
        cv2.imshow("Wear the Mask!", frame)
