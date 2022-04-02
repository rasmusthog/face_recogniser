import numpy as np
import os
import shutil

import cv2


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

label = input('Please enter name: ')

path = os.path.join('data/training/', label)

if not os.path.isdir(path):
    os.makedirs(path)

else:
    rewrite = input('Data already exists. Rewrite? [y/n] ')

    if rewrite:
        shutil.rmtree(path)
        os.makedirs(path)

    else:
        exit()



count = 0
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    cv2.imshow('frame', frame)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        filename = os.path.join(path, str(count+1).zfill(3))+'.jpg'

        cv2.imwrite(filename, roi_gray)

        count += 1

    if count >= 100:
        break


cap.release()
cv2.destroyAllWindows()