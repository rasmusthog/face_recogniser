import numpy as np
import cv2
import pickle


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('models/Rasmus_only.yml')

# Load labels:
with open('models/Rasmus_only.pickle', 'rb') as f:
    inv_labels = pickle.load(f)
    labels = {val:key for key, val in inv_labels.items()}

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recogniser.predict(roi_gray)

        if conf <= 25:
            text = labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (x,y), font, 2, (0,255,0), 5)

            print(conf)



    #frame = np.fliplr(frame)
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()