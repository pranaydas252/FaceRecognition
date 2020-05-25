import numpy
import cv2
import os

#face detection part
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_haar.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=3)
    return faces, gray_img

#labels for training data
def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirname, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print("img_path", img_path)
            print("ID",id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Not Loaded properly")
                continue

            faces_rect,gray_img = faceDetection(test_img)
            (x,y,w,h) = faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID

#Training classifier
def train_classifier(faces,facesID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,numpy.array(facesID))
    return face_recognizer

#Draw Rectangle
def draw_rect(test_img, face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

#Put Text
def put_text(test_img,label_name,x,y):
    cv2.putText(test_img,label_name,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)