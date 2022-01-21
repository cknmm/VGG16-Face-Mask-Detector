import cv2
#import tensorflow as tf
#from tensorflow import keras
import time, pathlib
import numpy as np
from keras.preprocessing import image
import os, time
import keras
import keras.engine
#import tensorflow.python.keras.engine

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#load the model:
model = keras.models.load_model("VGG16 Face Mask Detection")

#labels
labels = ["Incorrect_Way", "Without_Mask", "With_Mask"]

#func to predict in an image:
def predict(img_path):

    global model, labels
    
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    p = model.predict(img_batch)
    l = list(p[0])
    r = labels[l.index(max(l))]

    return (r, max(l))

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

if cap.isOpened():
    while True:

        #store starting time
        s = time.time()
        
        # Read the frame
        _, img = cap.read()
        img_dim = img.shape
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces):

            #write image frame
            cv2.imwrite("base_frame.png", img)

            #image paths for prediction
            to_predict = []
            #number of cropped images
            sub_cropped = 0
            
            #split the base frame into parts containing images
            base_0 = cv2.imread("base_frame.png")
            for (x, y, w, h) in faces:
                sub_cropped += 1
                cropped_image = base_0[y:y+h, x:x+w]

                f_name = "cropped_"  + str(sub_cropped) + ".png"
                cv2.imwrite(f_name, cropped_image)

                #store dimensions and name of the cropped parts
                details = {}
                details["name"] = f_name
                details["rect"] = (x, y, w, h)
                to_predict.append(details)

            #predict from those images:
            for i in to_predict:
                f_name = i["name"]
                x, y, w, h = i["rect"]

                #get prediction
                class_label, probability = predict(f_name)
                #text to put
                text_to_put = class_label + ", " + str(probability*100) + "%"

                #decide rectangle and text color
                if class_label == "Incorrect_Way":
                    color = (255, 255, 0)
                elif class_label == "With_Mask":
                    color = (0, 255, 0)
                elif class_label == "Without_Mask":
                    color = (255, 0, 0)
                    
                #draw rectangle and put text:
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, text_to_put, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2, cv2.LINE_AA)
            
            """# Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)"""
        
        #store end time
        e = time.time()

        #show prediction delay
        d = (e - s)*1000
        cv2.putText(img, str(round(d, 2)) + " ms", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
        
    # Release the VideoCapture object
    cap.release()
