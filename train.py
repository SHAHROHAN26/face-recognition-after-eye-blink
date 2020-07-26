# Import OpenCV2 for image processing
# Import os for file path
import cv2, os

# Import numpy for matrix calculation
import numpy as np

# Import Python Image Library (PIL)
from PIL import Image
from matplotlib import cm

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection
detector = cv2.CascadeClassifier("har.xml");

# Create method to get the images and label data
def getImagesAndLabels(path,i):

    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    # print(imagePaths) 
    
    # Initialize empty face sample
    faceSamples=[]
    
    # Initialize empty id
    ids = []

    # Loop all the file path
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        #PIL_img.show()

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')
        #faceSamples.append(img_numpy)
        # Get the image id
        id = i
        #print(id)

        # Get the face from the training images
        faces = detector.detectMultiScale(img_numpy)
        

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:    
            # Add the patch to the Axes
            #im = Image.fromarray(np.uint8(cm.gist_earth(img_numpy[y:y+h,x:x+w])*255))
            #im.show()
            
            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    # Pass the face array and IDs array
    return faceSamples,ids

# Get the faces and IDs
path='./datasets/'
datasets=[os.path.join(path,f) for f in os.listdir(path)]
faces_list=[]
id_list=[]
for i,x in enumerate(datasets):
    print(i)
    faces,ids = getImagesAndLabels(x,i)
    print(len(faces),len(ids))
    faces_list=np.append(faces_list,faces)
    id_list=np.append(id_list,ids)
    print(faces_list.shape,id_list.shape)
id_list=np.array(id_list,dtype='int32')
# Train the model using the faces and IDs
recognizer.train(faces_list, id_list)

# Save the model into trainer.yml
if not os.path.exists('trainer'):
    os.mkdir('trainer')
recognizer.save('trainer/trainer.yml')
