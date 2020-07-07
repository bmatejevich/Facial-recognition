#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: brianmatejevich
"""

import cv2
import os
picNum = 1

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# For a video, give the path; for a webcam, this is usually 0 or 1
vid = cv2.VideoCapture(0)

# Loop forever (until user presses q)
while True:
    # Read the next frame from the camera
    ret, frame = vid.read()

    # Check the return value to make sure the frame was read successfully
    if not ret:
        print('Error reading frame')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Copy the current frame for later display
    disp = frame.copy()
    
    # Detect faces in the gray image. (Lower numbers will detect more faces.)
    # Parameters:
    #   scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
    #   minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    

    # loop through detected face rectangles. Each rectangle is a tuple of 4
    # numbers: the x and y position of the top-left corner and the width and
    # height of the face.
    faces = 0
    for (x, y, w, h) in face_rects:
        # Draw a rectangle around the detected face
        disp = cv2.rectangle(disp, (x, y), (x+w, y+h), (100, 200, 0), 2)
        if faces == 0:
            x1 = x
            y1 = y
            w1 = w
            h1 = h
            face1 = disp[y:y+h,x:x+w]
            faces = 1
        imageName = "{}.jpg".format(picNum)
            
        face1 = cv2.resize(face1,(150,150))
        face1 = cv2.cvtColor(face1,cv2.COLOR_BGR2GRAY)
        path = "Positive/train"
        cv2.imwrite(os.path.join(path , imageName),face1)
        picNum = picNum + 1
    if picNum == 100:
        break

    cv2.imshow("Press 'q' to quit", disp)

    # Get which key was pressed
    key = cv2.waitKey(1)

    # Keep looping until the user presses q
    if key & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()