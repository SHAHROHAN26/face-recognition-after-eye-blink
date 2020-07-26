# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
name=input('Enter the name of whose photos are being stored: ');
# construct the argument parser and parse the arguments
ap=argparse.ArgumentParser()
args=vars(ap.parse_args())

#load haar cascade for face detection from disk
detector=cv2.CascadeClassifier('har.xml')
if not os.path.exists('datasets'):
        os.mkdir('datasets')
if not os.path.exists('datasets/'+name):
        os.mkdir('datasets/'+name)
photo_len=len(os.listdir('datasets/'+name))
print("Starting video stream..")
vs=VideoStream(src=0).start()

time.sleep(2.0)
total=0

#Video stream loop
while True:
	frame=vs.read()
	orig=frame.copy()
	frame=imutils.resize(frame,width=1080)
	#orig=frame.copy()
	#orig=cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)

	#detect faces in grayscale frame
	rects=detector.detectMultiScale(
	cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),scaleFactor=1.1,
	minNeighbors=5,minSize=(30,30))

	for (x,y,w,h) in rects:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	
	cv2.imshow("Frame",frame)
	key=cv2.waitKey(1) & 0xFF
                
	if key==ord('c'):
		p=os.path.sep.join(['datasets/'+name,"{}.png".format(str(total+photo_len).zfill(5))])
		cv2.imwrite(p,orig)
		total+=1
	
	elif key==ord('q'):
		break
print("{} face images stored".format(total))
print("cleaning up...")
cv2.destroyAllWindows()
vs.stop()
