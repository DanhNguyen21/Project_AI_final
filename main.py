import cv2
from mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model
import cvlib as cv
from time import sleep
cam=cv2.VideoCapture(0)
class_names_gender=['Male','Female']
model_gender=load_model('gender.h5')
model_age=load_model('age.h5')
while True:
    Ok,frame=cam.read()
    faces, confidence = cv.detect_face(frame)
    for i,f in enumerate(faces):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        cv2.rectangle(frame,(startX,startY),(endX,endY),(255,0,0),3)
        img_cat=frame[startY:endY,startX:endX]
        img_gender=np.resize(img_cat, (100,100,3))
        img_gender=img_gender.astype('float32')
        img_gender/=255
        output_gender=class_names_gender[np.argmax(model_gender.predict(img_gender.reshape(-1,100,100,3)))]
        img_age=np.resize(img_cat, (200,200,3))
        img_age=img_age.astype('float32')
        img_age/=255
        output_age=int(model_age.predict(img_age.reshape(-1,200,200,3)))
        col = (0,255,0)
        cv2.putText(frame, str(output_gender)+","+str(output_age),(startX,startY),cv2.FONT_HERSHEY_SIMPLEX,1,col,2)
    cv2.imshow('Project_AI',frame)
    sleep(0.01)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break;
cam.release()
cv2.destroyAllWindows()