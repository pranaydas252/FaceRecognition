import numpy as numpy
import cv2
import os
import faceRecognition as fr

test_img = cv2.imread(r"path\to\testing_image.jpg") #Path to testing image

faces_detected,gray_img = fr.faceDetection(test_img)
print("Face Detected: ",faces_detected)

#Begin training
faces,facesID = fr.labels_for_training_data(r"path\to\dataset") #Path to Dataset folder
face_recognizer = fr.train_classifier(faces,facesID)
face_recognizer.save(r'path\to\trainingData.yml') #Path to trainingData.yml

name={0:'Bill Gates'} #Add new face ID's accordingly

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+w,x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    print(label)
    print(confidence)
    fr.draw_rect(test_img,face)
    predict_name = name[label]
    fr.put_text(test_img,predict_name,x,y)

resized_img = cv2.resize(test_img,(800,700))

cv2.imshow("Face Detected: ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
