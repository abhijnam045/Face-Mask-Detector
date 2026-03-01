import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mask_detector.h5")

# Load face detection
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        # Convert BGR to RGB (VERY IMPORTANT)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (128,128))
        face = face / 255.0
        face = np.reshape(face, (1,128,128,3))

        prediction = model.predict(face)[0][0]
        if prediction > 0.5:
            label = f"Without Mask {prediction*100:.2f}%"
            color = (0,0,255)
        else:
            label = f"With Mask {(1-prediction)*100:.2f}%"
            color = (0,255,0)
            
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
