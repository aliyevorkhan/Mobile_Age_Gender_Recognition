import cv2 as cv
import argparse
import time
import math
import random

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.config import Config
from kivy.uix.togglebutton import ToggleButton

Config.set('graphics', 'width', '360')
Config.set('graphics', 'height', '600')


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                         (255, 0, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(
    description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument(
    '--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-3)', '(4-7)', '(8-14)', '(15-20)',
           '(21-35)', '(36-45)', '(46-56)', '(57-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

cap = cv.VideoCapture(0)
padding = 20

class MainWindow(BoxLayout):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)

class CamApp(App):

    def no_face(self):
        self.lblOutput.text = "NO face detected, checking next frame"
       
    def capture(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.camera.export_to_png("IMG_{}.png".format(timestr))

    def output(self, gender, genderPreds, age, agePreds):
        self.lblOutput.text = "Gender : {}, conf = {:.3f}".format(
            gender, genderPreds[0].max()) + "\n" + "Age : {}, conf = {:.3f}".format(
            age, agePreds[0].max())

    def build(self):
        self.camera = Camera()
        
        superLayout = MainWindow(orientation='vertical')
        verticalLayout = MainWindow(orientation='vertical')

        verticalLayout.add_widget(self.camera)
        self.lblOutput = Label(text="initializing...",
                               font_size=8,
                               color=(0, 0, 1, 1),
                               size=(50, 50),
                               size_hint=(.4, .1))

        verticalLayout.add_widget(self.lblOutput)

        superLayout.add_widget(verticalLayout)
        self.capture = cap
        Clock.schedule_interval(self.update, 1.0/33.0)
        return superLayout

    def update(self, dt):

        ret, frame = self.capture.read()
        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            self.no_face()

        for bbox in bboxes:
            face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1), max(
                0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            #print("Age Output : {}".format(agePreds))
            self.output(gender, genderPreds,age,agePreds)
            #print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

        
        buf1 = cv.flip(frameFace, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera.texture = texture1
        
if __name__ == '__main__':
    CamApp().run()
