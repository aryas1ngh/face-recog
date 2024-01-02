import cv2
import numpy as np
from PIL import Image
import os

cascade_dir = './cascades/'
data_dir = './data/'
trainer_dir = './trainer/'


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cascade_dir+'haarcascade_frontalface_default.xml');

def getImagesAndLabels(data_dir):
    imagePaths = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("Training faces...")
faces,ids = getImagesAndLabels(data_dir)
recognizer.train(faces, np.array(ids))

# save model
recognizer.write(trainer_dir+'lbph_trainer.yml') 

print("{0} faces trained.".format(len(np.unique(ids))))
