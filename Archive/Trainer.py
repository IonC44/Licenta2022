# -*- coding: utf-8 -*-

"""
Created on Sat Aug  6 07:10:42 PM 2022

@author: ion.ceparu
"""

import matplotlib
matplotlib.use("Agg")

from config import config
from preprocesare import image_to_array
from Trainer_Info import epochcallback
from Trainer_Info import trainmonitor
from IO import hdf5_to_data
import VGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
import keras.backend as BK
import argparse
import os

ap = argparse.ArgumentParser()

ap.add_argument("-c", "--checkpoints", required = True,
                 help = "path to output checkpoint")
ap.add_argument("-m", "--model", type = str, help = "path to model checkpoint")
ap.add_argument("-s", "--epoch_start", type = int, default = 0, 
                help = "epoch at which the training will start")
args = vars(ap.parse_args())

#build the training/tesitng image generators for data augmentation and
#then inialize the preproccesor

train_augm = ImageDataGenerator(rotation_range = 10, zoom_range = 0.1, 
                                horizontal_flip = True, rescale = 1/255.0, 
                                fill_mode = "nearest", width_shift_range = 0.2,
                                height_shift_range = 0.2)
valid_augm = ImageDataGenerator(rescale = 1 / 255.0)
itapp = image_to_array.ImageToArray()

train_hdf5 = hdf5_to_data.HDF5_To_Data(config.TRAIN_DATA_HDF5, config.DIM_BATCH,
                                        preprocesor = [itapp],
                                       n_labels = config.N_LABELS)
valid_hdf5 = hdf5_to_data.HDF5_To_Data(config.VALID_DATA_HDF5, config.DIM_BATCH,
                                        preprocesor = [itapp],
                                       n_labels = config.N_LABELS)

#if we have no model supplied, we build one
if args["model"] is None:
    print("######### [INFO] - Creating Model #########")
    model = VGGNet.VGGNet.build(w = 48, h = 48, d = 1, labels = config.N_LABELS)
    optimization_method = SGD(learning_rate = 1e-3, momentum = 0.9, nesterov  = True)
    model.compile(loss = "categorical_crossentropy", optimizer = optimization_method, metrics = ["accuracy"])
else:
    #load the model checkpoint from the disk
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])
    optimization_method = Adam(learning_rate = 1e-4 )
    model.compile(loss = "categorical_crossentropy", optimizer = optimization_method, metrics = ["accuracy"])


#print the info and save it
fig_path = os.path.sep.join([config.OUTPUT_PATH,
                            "vggnet_emotion.png"])
json_path = os.path.sep.join([config.OUTPUT_PATH,
                             "vggnet_emotion.json"])

callbacks = [epochcallback.EpochCallback(args["checkpoints"], each = 1, 
                                              start_at = args["epoch_start"]),
             trainmonitor.TrainMonitor(fig_path, json_path = json_path, 
                                       start_at = args["epoch_start"]),
             ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                           patience = 5, min_lr = 1e-5, verbose = 1)]

#Finally train the network
model.fit(
    train_hdf5.to_data(),
    steps_per_epoch = train_hdf5.n_imag,
    validation_data = valid_hdf5.to_data(),
    epochs = 10, 
    callbacks = callbacks,
    validation_steps = valid_hdf5.n_imag,
    #max_queue_size = config.DIM_BATCH * 2,
     verbose = 1)

#close the databases
train_hdf5.close()
valid_hdf5.close()
