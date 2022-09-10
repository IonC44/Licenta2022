# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:52:42 2022

@author: ion.ceparu
"""

import os
import argparse
import random

ap = argparse.ArgumentParser()

ap.add_argument("-dp", "--datapath", required = True,
                 help = "path to Data folder")
ap.add_argument("-pp", "--projectpath", required = True,
                 help = "path to project folder")
ap.add_argument("-m", "--model", type = str, required = True, 
                help = "path to model checkpoint")
args = vars(ap.parse_args())


exec(open(os.path.sep.join([args["projectpath"], "Scripts/Useful_Scripts.py"])).read())

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

#Categories for FER
categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

#Get ImageDataGenerator Object for training Data
train_IDG = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None
)

#Get the images
train_gen = train_IDG.flow_from_directory(
    directory=os.path.sep.join([args["datapath"],'Data_Images/Training']),
    target_size=(224, 224),
    color_mode='rgb',
    classes=categories,
    class_mode='categorical',
    batch_size=3500,
    shuffle=False
) 

#Get the first batch and release memory
aux = train_gen.next()
rgb_train_images = aux[0]
train_labels = aux[1]

del aux
gc.collect()

#Get the validation and testing images
path_val = os.path.sep.join([args["datapath"],"Data/data_val.csv"])
path_test = os.path.sep.join([args["datapath"],"Data/data_test.csv"])
rgb_valid_labels, valid_labels = csv_to_rgb_list(path_val)
rgb_test_images, test_labels = csv_to_rgb_list(path_test)

model = tf.keras.models.load_model(args["model"])

#Results
print("ResNet34 Performance:")

#For training 
predict_train = model.predict(rgb_train_images)
loss_train = tf.keras.losses.CategoricalCrossentropy()(train_labels, predict_train).numpy()
accuracy_train = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(train_labels, predict_train).numpy()
print('Train Loss: ', '%.10f' % loss_train)
print('Train accuracy: ', '%.10f' % accuracy_train)

#For validation
predict_val = model.predict(rgb_valid_labels)
accuracy_val = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(valid_labels, predict_val).numpy()
print('Val accuracy: ', '%.10f' % accuracy_val)

#For testing
predict_test = model.predict(rgb_test_images)
accuracy_test = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(test_labels, predict_test).numpy()
print('Test accuracy: ', '%.10f' % accuracy_test)




