# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:13:20 2022

@author: ion.ceparu
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import time


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path the face cascade ")
ap.add_argument("-m", "--model", required=True,
               help="path to model")
ap.add_argument("-v", "--video",
                help="path to video")
args = vars(ap.parse_args())

#Load the face detector Haar
detector = cv2.CascadeClassifier(args["cascade"])

#Load the model
model = tf.keras.models.load_model(args["model"])

categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']


if args["video"] is None:
    cam = cv2.VideoCapture(0)
else:
    cam = cv2.VideoCapture(args["video"])

#Infinite Loop
while True:
    
    cv2.ocl.setUseOpenCL(False)
    #take the frame
    (check, frame) = cam.read()
    
    if args.get("video") and not check:
        break
    
    #Resize the frame 
    frame = imutils.resize(frame,width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = detector.detectMultiScale(rgb, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    #Init a draw board and clone the frame
    draw = np.zeros((255,400,3), dtype="uint8")
    frame_2 = frame.copy()
    
    if len(faces) > 0:
        
        
        #Take the Face with the biggest Radius
        faces = sorted(faces,reverse=True,
                       key=lambda x: (x[2] - x[0])*(x[3] - x[1]))[0]
        (fx, fy, fw, fh) = faces
        
        #Extract the face 
        #fx - coordonate x where the face begins
        #fy - coordonate y where the face begins
        #fw - face width
        #fh - face height   
        #Extract the face
        face = rgb[fy : fy + fh, fx : fx + fw]
        face = cv2.resize(face, (224,224))
        face = img_to_array(face)
        #face = face.astype("float32")
        face = np.expand_dims(face, axis = 0)
               
        #get the prediction
        pred = model.predict(face)[0]
        label = categories[pred.argmax()]
        
        #draw the emotions and probablities
        for (i, (categ, p)) in enumerate(zip(categories, pred)):
             
             #the text which we we'll put on the screen
             text = "{}: {:.2f}%".format(categ, p * 100)
             
             #let's draw!
             px = int(p*400) #the final coordonate of the rectangle of the probability
             cv2.rectangle(draw, (5, (i*35) +5),
                           (px,(i*35) + 35), (138,43,226), -1)
             cv2.putText(draw, text, (10, (i * 35) + 23),
                         cv2.FONT_HERSHEY_DUPLEX, 0.55,
                         (255, 255, 255), 2)
             
             cv2.putText(frame_2, label, (fx, fy-10), 
                         cv2.FONT_HERSHEY_DUPLEX, 0.55, (138,43,226), 2)
             cv2.rectangle(frame_2, (fx, fy), (fx+fw,fy+fh), (138,43,226) , 2)
        cv2.imshow("Fata", frame_2)
        cv2.imshow("Probabilitati", draw)
        
        #exit with q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(1/60)
        
cam.release()
cv2.destroyAllWindows()
        
        
        
        
        