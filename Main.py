# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:10:35 2022

@author: ion.ceparu
"""


import json
with open('Config/Config.json') as json_file:
    config = json.load(json_file)

# IMPORTS
import random
import argparse
import os
from Scripts.Useful_Scripts import *
from Model import resNet
from Trainer_Info.trainmonitor import TrainMonitor

# Constants
CATEGORIES = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
BATCH_SIZE = 128

# ARGUMENT PARSER
ap = argparse.ArgumentParser()

ap.add_argument("-dp", "--datapath", required = True, help = "path to Data folder")
ap.add_argument("-pp", "--projectpath", required = True, help = "path to project folder")
ap.add_argument("-o", "--outputmodel", required = True, help = "path to model output")
ap.add_argument("-m", "--model", type = str, help = "path to model checkpoint")
args = vars(ap.parse_args())

# DATA LOADING
columns = [i for i in range(2049) if i != 1]

train_dataset = tf.data.TextLineDataset(os.path.sep.join([args["datapath"], "data_test.csv"])) \
    .skip(1) \
    .map(parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.TextLineDataset(os.path.sep.join([args["datapath"], "data_val.csv"])) \
    .skip(1) \
    .map(parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# PREPROCCESING
# train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1/255,
#     samplewise_center = True,
#     samplewise_std_normalization = True,
#     rotation_range=config['rotation_range'],
#     width_shift_range=config['width_shift_range'],
#     height_shift_range=config['height_shift_range'],
#     zoom_range=config['zoom_range'],
#     horizontal_flip=config['horizontal_flip'],
# )


# train_generator = train_data_gen.flow_from_directory(
#     directory=path_train,
#     target_size=(224, 224),
#     color_mode='rgb',
#     classes=Categories,
#     class_mode='categorical',
#     batch_size=config['batch_size'],
# )

# valid_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1/255,
#     samplewise_center = True,
#     samplewise_std_normalization = True)


# valid_generator = train_data_gen.flow_from_directory(
#     directory=path_val,
#     target_size=(224, 224),
#     color_mode='rgb',
#     classes=Categories,
#     class_mode='categorical',
#     batch_size=config['batch_size']
# )

# # LOAD_MODEL
if args["model"] is None:
  model = resNet(train_dataset.element_spec[0].shape[1:], 2, num_classes=len(CATEGORIES))
else:
  model = tf.keras.models.load_model(args["model"])

print(model.summary())

# # ----------------------------TRAINING-----------------------------------------


if config['optimizer'] == "Adam":
    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], name='Adam')
elif config['optimizer'] == "SGD":
    opt = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'], name='SGD')
else:
    raise ValueError("No optimizer found!")




model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['SparseCategoricalCrossentropy'])

# callback = TrainMonitor(os.path.sep.join([args["outputmodel"], 'Train_History']),
#                           os.path.sep.join([args["outputmodel"], 'Train_History']), 
#                         )

history = model.fit(
    x=train_dataset,
    epochs=300,
    verbose=1,
    # callbacks=[callback],
    validation_data=validation_dataset
)
np.save(os.path.sep.join([args["outputmodel"], 'history.npy']),history.history)



