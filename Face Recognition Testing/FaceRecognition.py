import cv2
import numpy as np

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("model_wajah.xml")
label_map = np.load("label_map.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        wajah = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(wajah)

        if confidence < 100:  
            nama = label_map[label]
            text = f"{nama} ({confidence:.2f})"
            color = (0, 255, 0)
        else:
            text = "Tidak Dikenali"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Pengenalan Wajah", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()