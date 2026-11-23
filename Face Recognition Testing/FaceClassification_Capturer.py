import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

dataset_path = "dataset"
nama_user = input("Masukkan nama Anda: ")
user_path = os.path.join(dataset_path, nama_user)

if not os.path.exists(user_path):
    os.makedirs(user_path)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        wajah = gray[y:y + h, x:x + w]
        cv2.imwrite(f"{user_path}/{count}.jpg", wajah)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Ambil Data Wajah", frame)

    if count >= 100:  
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Data wajah {nama_user} tersimpan di folder {user_path}")