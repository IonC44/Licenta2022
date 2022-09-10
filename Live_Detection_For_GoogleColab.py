from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
               help="path to model")
ap.add_argument("-p", "--path", required = True,
                help="path to project folder")

args = vars(ap.parse_args())

#Load the face detector Haar
detector = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
#Load the model
model = tf.keras.models.load_model(args["model"])

categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

exec(open(os.path.sep.join([args["path"], "Scripts/Java_Script_Video.py"])).read())

# start streaming video from webcam
video_stream()
# label for video
label_html = 'Capturing...'
# initialze bounding box to empty
bbox = ''
count = 0 
while True:
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break

    # convert JS response to OpenCV Image
    img = js_to_image(js_reply["img"])

    # create transparent overlay for bounding box
    bbox_array = np.zeros([480,640,4], dtype=np.uint8)

    # get face region coordinates
    faces = detector.detectMultiScale(img, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    
    # get face bounding box for overlay
    for (fx,fy,fw,fh) in faces:
      face = img[fy : fy + fh, fx : fx + fw]
      face = cv2.resize(face, (224,224))
      face = img_to_array(face)
      face = np.expand_dims(face, axis = 0)
               
      #get the prediction
      pred = model.predict(face)[0]
      label = categories[pred.argmax()]

      bbox_array = cv2.rectangle(bbox_array, (fx, fy), (fx+fw,fy+fh), (138,43,226) , 2)
      bbox_array = cv2.putText(bbox_array, label, (fx, fy-10), 
                         cv2.FONT_HERSHEY_DUPLEX, 0.55, (138,43,226), 2)


    bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
    # convert overlay of bbox into bytes
    bbox_bytes = bbox_to_bytes(bbox_array)
    # update bbox so next frame gets new overlay
    bbox = bbox_bytes
