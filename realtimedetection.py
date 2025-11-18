import cv2
from keras.models import load_model
import numpy as np

# Load model directly
model = load_model("facialemotionmodel.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)
labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}

while True:
    ret, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))

        img = extract_features(roi)
        pred = model.predict(img)
        label = labels[pred.argmax()]

        cv2.rectangle(im, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(im, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Emotion Detection", im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
