from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

def eye_aspect_ratio(eye):
    # vertical eye landmarks
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    # horizontal eye landmarks
    C=dist.euclidean(eye[0],eye[3])
    # compute the eye aspect ratio
    ear=(A+B)/(2.0*C)
    #return the eye aspect ratio
    return ear

# passing arguments for detection
ap= argparse.ArgumentParser()
ap.add_argument("-v","--video",type=str,default="",help="path to video file")
args=vars(ap.parse_args())

EYE_AR_THRESH=0.27
EYE_AR_CONSEC_FRAMES=3

COUNTER=0
data_list=os.listdir('./datasets')
TOTAL=[0 for i in range(len(data_list))]

print("facial landmark predictor...")

detector=dlib.get_frontal_face_detector()

predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



# start the video stream thread
print("webcam starting.........")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

#Face_recognition
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath='har.xml'
faceCascade=cv2.CascadeClassifier(cascadePath)
font=cv2.FONT_HERSHEY_SIMPLEX

while True:
    if fileStream and not vs.more():
        break
    frame=vs.read()
    frame=imutils.resize(frame,width=1080)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces=faceCascade.detectMultiScale(gray,1.2,5)

    rects=detector(gray,0)

#get ID of face but can't give name untill 2 eye blink detected 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),4)

        ID=recognizer.predict(gray[y:y+h,x:x+w])
        print(ID)
        '''if (int(ID[0])<len(data_list)) and (ID[1]<45):
            name=data_list[ID[0]]
        else:
            name="Unknown"
        print(name)

        cv2.rectangle(frame,(x-22,y-90),(x+w+22,y-22),(0,255,0),-1)
        cv2.putText(frame,str(name),(x,y-40),font,1,(255,255,255),3)'''
    for rect in rects:
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)

        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEAR=eye_aspect_ratio(leftEye)
        rightEAR=eye_aspect_ratio(rightEye)

        ear=(leftEAR+rightEAR)/2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
        # if the eyes were closed for a sufficient number of times
        # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES and (ID[1]<45):
                TOTAL[ID[0]] += 1
             # reset the eye frame counter
            COUNTER = 0

# face recognizer only after 2 eye blink
    
        if max(TOTAL) > 1:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),4)

                ID=recognizer.predict(gray[y:y+h,x:x+w])
                print(ID)
                if (int(ID[0])<len(data_list)) and (ID[1]<45):
	                name=data_list[ID[0]]
                else:
	                name="Unknown"
                print(name)

                cv2.rectangle(frame,(x-22,y-90),(x+w+22,y-22),(0,255,0),-1)
                cv2.putText(frame,str(name),(x,y-40),font,1,(255,255,255),3)
# draw the total number of blinks on the frame along with
# the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    if max(TOTAL)>3:
        print("***************end**************")
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
