import cv2
import os
from PIL import Image
import numpy as np
import pickle


BASE_DIR = 'data/training'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()
model_name = input('Enter model name: ')


if os.path.isfile(f'models/{model_name}.yml'):
    overwrite = input(f'Model "{model_name}" already exists. Overwrite? [y/n] ').lower()
    if overwrite in ['n', 'no']:
        print('Exiting...')
        exit()
    else:
        print('Overwriting...')

training_sets = [subdir for subdir in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, subdir))]

x_train = []
y_labels = []
label_map = {}
current_id = 0

for set in training_sets:
    print(f'Loading set: {set}')
    set_dir = os.path.join(BASE_DIR, set)

    if not set in label_map:
        label_map[set] = current_id
        id_ = current_id
        current_id += 1
        

    imgs = [img for img in os.listdir(set_dir) if os.path.isfile(os.path.join(set_dir, img)) and img.endswith('.png') or img.endswith('.jpg')]
    
    for img in imgs:
        print(f'... Processing image: {img}')

        img_path = os.path.join(set_dir, img)

        pil_image = Image.open(img_path).convert('L')
        size = (550, 500)
        final_image = pil_image.resize(size, Image.ANTIALIAS)
        img_array = np.array(final_image, 'uint8')

        faces = face_cascade.detectMultiScale(img_array, 1.3, 4)

        for (x, y, w, h) in faces:
            roi = img_array[y:y+h, x:x+w]

            x_train.append(roi)
            y_labels.append(id_)



with open(f'models/{model_name}.pickle', 'wb') as f:
    pickle.dump(label_map, f)

recogniser.train(x_train, np.array(y_labels))
recogniser.save(f'models/{model_name}.yml')



