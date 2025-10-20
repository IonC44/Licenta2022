# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:10:35 2022

@author: ion.ceparu
"""

# IMPORTS
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import random
from classification_models.tfkeras import Classifiers
import argparse
import os
from Scripts.Useful_Scripts import *


# Function for a layer of ResNet
def resnet_layer(X, stage, num_filters=16, kernel_size=3, activation='relu', batch_normalization=True):

    # Naming convention
    conv_name = 'res' + str(stage) + '_conv'
    batch_name = 'res' + str(stage) + '_bn'
    activation_name = 'res' + str(stage) + '_act'

    # First Conv2D layer
    Y = layers.Conv2D(num_filters, kernel_size, strides=(1, 1), padding='same', 
                                  activation=None, use_bias=True, name=conv_name + '1')(X)
    Y = layers.BatchNormalization(axis=-1,name=batch_name + '1')(Y)
    Y = layers.Activation(activation='relu', name=activation_name + '1')(Y)
    # Second Conv2D layer
    Y = layers.Conv2D(num_filters, kernel_size, strides=(1, 1), padding='same', 
                                  activation=None, use_bias=True, name=conv_name + '2')(Y)
    Y = layers.BatchNormalization(axis=-1,name=batch_name + '2')(Y)
    # Add identity
    Y = layers.Add()([Y, X])
    Y = layers.Activation(activation='relu',name=activation_name + '2')(Y)

    return Y

# Function for resNet Model
def resNet(input_shape, depth, num_classes=10):

    X_input = layers.Input(shape=input_shape)

    num_ResNet_Blocks = depth
    for i in range(num_ResNet_Blocks):
        X = resnet_layer(X_input, num_filters=16 * depth, stage=i)
    
    X = layers.Flatten()(X)
    X = layers.Dense(num_classes, activation='softmax', name='fc' + str(num_classes))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model

