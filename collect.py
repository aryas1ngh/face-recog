import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
cascade_dir = './cascades/'
data_dir = './data/'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

face_detector = cv2.CascadeClassifier(cascade_dir+'haarcascade_frontalface_default.xml')
face_id = input('\n Enter User ID :')
print("\n Initializing face capture. Look at the camera and wait ...")

count = 0
while(True):
    ret, img = cam.read()
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # save img
        cv2.imwrite(data_dir+"/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 100:
         break

print("\nExiting...")
cam.release()
cv2.destroyAllWindows()