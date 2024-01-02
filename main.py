import cv2
import numpy as np
import os 
import time


#dir
data_dir = './data/'
cascade_dir = './cascades/haarcascade_frontalface_default.xml'
trainer_dir = './trainer/'
recognized_names=[]

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_dir+'lbph_trainer.yml')
face_cascade = cv2.CascadeClassifier(cascade_dir)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['arya', 'aditya', 'messi'] 


cam = cv2.VideoCapture(0)
cam.set(3, 640) #w
cam.set(4, 480) #h

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
        time.sleep(0.2)
        if id not in recognized_names:
            recognized_names.append(id)
                
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("Present Students  :", recognized_names)
cam.release()
cv2.destroyAllWindows()