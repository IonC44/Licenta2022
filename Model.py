# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:10:35 2022

@author: ion.ceparu
"""

# IMPORTS
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from Scripts.Useful_Scripts import *
import copy


# Function for a layer of ResNet
def resnet_layer(X, stage, num_filters=16, kernel_size=3, activation='relu', batch_normalization=True):

    # Naming convention
    conv_name = 'res' + str(stage) + '_conv'
    batch_name = 'res' + str(stage) + '_bn'
    activation_name = 'res' + str(stage) + '_act'
    add_name = 'res' + str(stage) + '_add'

    # First Conv2D layer
    Y = layers.Conv2D(num_filters, kernel_size, strides=(1, 1), padding='same', 
                                  activation=None, use_bias=True, name=conv_name + '1')(X)
    Y = layers.BatchNormalization(axis=-1,name=batch_name + '1')(Y)
    Y = layers.Activation(activation='relu', name=activation_name + '1')(Y)
    # Second Conv2D layer
    Y = layers.Conv2D(num_filters, kernel_size, strides=(1, 1), padding='same', 
                                  activation=None, use_bias=True, name=conv_name + '2')(Y)
    Y = layers.BatchNormalization(axis=-1,name=batch_name + '2')(Y)
    # Before adding, check that the channels are the same
    if X.shape[-1] != 1 & X.shape[-1] != num_filters:
      X = layers.Conv2D(num_filters, (1, 1), padding='same', name=conv_name + 'Convert')(X)
      X = layers.BatchNormalization(axis=-1, name=batch_name + 'Convert')(X)
    
    # Add identity
    Y = layers.Add(name=add_name)([Y, X])
    Y = layers.Activation(activation='relu',name=activation_name + '2')(Y)

    return Y

# Function for resNet Model
def resNet(input_shape, depth, num_classes):

    X_input = layers.Input(shape=input_shape)
    X = resnet_layer(X_input, num_filters=16, stage=0)

    for i in range(depth - 1):
        X = resnet_layer(X, num_filters=16 * (2 ** (i + 1)), stage=(i + 1))
    
    X = layers.Flatten()(X)
    X = layers.Dense(num_classes, activation='softmax', kernel_regularizer='l2', name='fc' + str(num_classes))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model

