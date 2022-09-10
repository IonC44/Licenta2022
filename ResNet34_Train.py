# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:10:35 2022

@author: ion.ceparu
"""

import random
from classification_models.tfkeras import Classifiers
import argparse
import os
from keras.utils.vis_utils import plot_model

ap = argparse.ArgumentParser()

ap.add_argument("-dp", "--datapath", required = True,
                 help = "path to Data folder")
ap.add_argument("-pp", "--projectpath", required = True,
                 help = "path to project folder")
ap.add_argument("-o", "--outputmodel", required = True,
                 help = "path to model output")
ap.add_argument("-m", "--model", type = str, help = "path to model checkpoint")
args = vars(ap.parse_args())


if not os.path.isdir(os.path.sep.join([args["outputmodel"], 'Models'])):
    os.mkdir(os.path.sep.join([args["outputmodel"], 'Models']))

exec(open(os.path.sep.join([args["projectpath"], "Scripts/Useful_Scripts.py"])).read())

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(5678)
random.seed(9101112)
np.random.seed(131415)

path_val = os.path.sep.join([args["datapath"],"Data/data_val.csv"])
path_train = os.path.sep.join([args["datapath"],'Data_Images/Training'])

rgb_valid_images, valid_labels = csv_to_rgb_list(path_val)
Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

train_IDG = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
    rotation_range=25,
    horizontal_flip=True
)

train_gen = train_IDG.flow_from_directory(
    directory=path_train,
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=44,
)


if args["model"] is None:
  ResNet34, preprocess_input = Classifiers.get('resnet34')
  base_model = ResNet34(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

  x = base_model.input
  y = preprocess_input(x)
  y = base_model(y)
  print(base_model.summary())
  plot_model(base_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  y = tf.keras.layers.GlobalAveragePooling2D()(y)
  y = tf.keras.layers.Dense(units=2048, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=3))(y)
  y = tf.keras.layers.Dropout(rate=0.5)(y)
  y = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=4))(y)
  y = tf.keras.layers.Dropout(rate=0.5)(y)
  y = tf.keras.layers.Dense(units=7, activation='softmax')(y)
  model = tf.keras.Model(inputs=x, outputs=y)
else:
  model = tf.keras.models.load_model(args["model"])

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-4,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False,
                               name='Adam')

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=25, restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint(os.path.sep.join([args["outputmodel"], 'Models/Pre']), monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
callback3 = tf.keras.callbacks.ReduceLROnPlateau(mointor='val_categorical_accuracy', factor=0.1, verbose=1, patience = 8, min_lr = 1e-5)   

history = model.fit(
    train_gen,
    epochs=200,
    verbose=1,
    callbacks=[callback1, callback2, callback3],
    validation_data=(rgb_valid_images, valid_labels)
)



train_IDG = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0,
    zoom_range=0.15,
    horizontal_flip=True
)


train_gen = train_IDG.flow_from_directory(
    directory=path_train,
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=64,
)


opt = tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', 
                                             patience=15, restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint(os.path.sep.join([args["outputmodel"], 'Models/Pre_2']), 
                                               monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
callback3 = tf.keras.callbacks.ReduceLROnPlateau(mointor='val_categorical_accuracy', factor=0.1, verbose=1, patience = 8, min_lr = 1e-5)   

history2 = model.fit(
    train_gen,
    epochs=200,
    verbose=1,
    callbacks=[callback1, callback2, callback3],
    validation_data=(rgb_valid_images, valid_labels)
)


tf.random.set_seed(1115)
random.seed(1115)
np.random.seed(1115)


train_IDG = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
)


train_gen = train_IDG.flow_from_directory(
    directory=path_train,
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=64
)



opt = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint(os.path.sep.join([args["outputmodel"], 'Models/Pre_3']), 
                                               monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
callback3 = tf.keras.callbacks.ReduceLROnPlateau(mointor='val_categorical_accuracy', factor=0.1, verbose=1, patience = 5, min_lr = 1e-5)   


history3 = model.fit(
    train_gen,
    epochs=200,
    verbose=1,
    callbacks=[callback1, callback2,callback3],
    validation_data=(rgb_valid_images, valid_labels)
)

