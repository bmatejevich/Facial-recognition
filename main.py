#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: brianmatejevich
"""
import cv2, os, numpy as np

negativeTrain = "Negative/train"
positiveTrain = "Positive/train"


# load in a folder of images and put them in a list
def load(folder):
    images = []
    for picture in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, picture))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (150, 150))
        if img is not None:
            images.append(img)
    return images


def makeList(folder1, folder2):
    labels = []
    images1 = load(folder1)
    images2 = load(folder2)
    images = images1 + images2
    for picture in os.listdir(folder1):
        labels.append(0)
    for picture in os.listdir(folder2):
        labels.append(1)
    return images, labels


# trian the model
def train():
    facesTrain, labelsTrain = makeList(negativeTrain, positiveTrain)
    labelsTrain = np.array(labelsTrain)
    model = cv2.face.EigenFaceRecognizer_create()
    print("training...")
    model.train(facesTrain, labelsTrain)
    m = model.getMean()
    m = np.reshape(m, (150, 150))
    m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return model


# main method
def go():
    if liveDemo == True:
        model = train()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        vid = cv2.VideoCapture(0)
        while True:
            ret, frame = vid.read()
            if not ret:
                print('Error reading frame')
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            disp = frame.copy()
            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in face_rects:
                face = disp[y:y + h, x:x + w]
                face = cv2.resize(face, (150, 150))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                pred_label, confidence = model.predict(face)
                if pred_label == 1:
                    disp = cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 125), 2)
                if pred_label == 0:
                    disp = cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow('Press "q" to quit', disp)
            key = cv2.waitKey(1)
            # Keep looping until the user presses q
            if key & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()


liveDemo = True
go()
