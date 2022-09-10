# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:10:35 2022

@author: ion.ceparu
"""
#IMPORTS
import random
from classification_models.tfkeras import Classifiers
import argparse
import os
from keras.utils.vis_utils import plot_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()


#ARGUMENT PARSER
ap = argparse.ArgumentParser()

ap.add_argument("-dp", "--datapath", required = True,
                 help = "path to Data folder")
ap.add_argument("-pp", "--projectpath", required = True,
                 help = "path to project folder")
ap.add_argument("-o", "--outputmodel", required = True,
                 help = "path to model output")
ap.add_argument("-m", "--model", type = str, help = "path to model checkpoint")
args = vars(ap.parse_args())

exec(open(os.path.sep.join([args["projectpath"], "Scripts/Useful_Scripts.py"])).read())
exec(open(os.path.sep.join([args["projectpath"], "Trainer_Info/trainmonitor.py"])).read())
exec(open(os.path.sep.join([args["projectpath"], "Trainer_Info/epochcallback.py"])).read())

#MAKE FOLDER
# if not os.path.isdir(os.path.sep.join([args["outputmodel"], 'Models'])):
#     os.mkdir(os.path.sep.join([args["outputmodel"], 'Models']))

#PATH SETUP
path_val = os.path.sep.join([args["datapath"],"Data_Images/Validation"])
path_train = os.path.sep.join([args["datapath"],'Data_Images/Training'])

Categories = ['Anger', 'Fear','Disgust', 'Happiness', 'Sadness', 'Surprise', 'Neutral', 'Unkown']


#LOAD_MODEL
if args["model"] is None:
    
    
  ResNet34, preprocess_input = Classifiers.get('seresnet34')
  base_model = ResNet34(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

  x = base_model.input
  y = preprocess_input(x)
  y = base_model(y)
  #print(base_model.summary())
  #plot_model(base_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  y = tf.keras.layers.GlobalAveragePooling2D()(y)
  y = tf.keras.layers.Dense(units=2048, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(y)
  y = tf.keras.layers.Dropout(rate=0.5)(y)
  y = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(y)
  y = tf.keras.layers.Dropout(rate=0.5)(y)
  y = tf.keras.layers.Dense(units=6, activation='softmax')(y)
  model = tf.keras.Model(inputs=x, outputs=y)
else:
  model = tf.keras.models.load_model(args["model"])

# ----------------------------TRAINING------SGD -------------------------------------




train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0.2,
    horizontal_flip=True
)


train_generator = train_data_gen.flow_from_directory(
    directory=path_train,
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=64,
)

valid_data_gen = tf.keras.preprocessing.image.ImageDataGenerator()


valid_generator = train_data_gen.flow_from_directory(
    directory=path_val,
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=64,
)

#Scheduler
def scheduler(epoch, lr):
    return lr*(10**(-int(epoch/30)))

opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum = 0.9, nesterov = True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=30, restore_best_weights=True)
callback2 = EpochCallback(args["outputmodel"], each = 10, start_at = 6)
callback3 = TrainMonitor(os.path.sep.join([args["outputmodel"], 'Train_History']),
                          os.path.sep.join([args["outputmodel"], 'Train_History']), 
                          start_at = 6)
callback4 = tf.keras.callbacks.LearningRateScheduler(scheduler)


history = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2, callback3],
    validation_data=valid_generator
)
np.save(os.path.sep.join([args["outputmodel"], 'history.npy']),history.history)



