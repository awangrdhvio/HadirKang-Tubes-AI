import cv2
import os
import numpy as np

dataset_path = "dataset"
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}

for label, user in enumerate(os.listdir(dataset_path)):
    user_path = os.path.join(dataset_path, user)
    label_map[label] = user
    
    for img_name in os.listdir(user_path):
        img_path = os.path.join(user_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        faces.append(img)
        labels.append(label)

face_recognizer.train(faces, np.array(labels))
face_recognizer.save("model_wajah.xml")

np.save("label_map.npy", label_map)

print("Model wajah berhasil dilatih dan disimpan!")