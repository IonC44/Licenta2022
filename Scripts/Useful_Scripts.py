# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 13:12:36 2022

@author: ion.ceparu
"""
import os
import tensorflow as tf
import csv
from PIL import Image
import numpy as np

import gc

def csv_to_list(file_name):
    data = open(file_name)
    read = csv.reader(data, delimiter=';')
    next(read)
    data = []
    for row in read:
        item = [row[0], row[2:]]
        data.append(item)
    images = np.zeros((len(data), 48, 48, 1), dtype='float32')
    labels = np.zeros((len(data)), dtype='float32')
    labels_binarized = np.zeros(shape=(len(data), 6), dtype='float32')
    for i in range(len(data)):
        images[i, :, :, :] = np.array(data[i][1]).reshape((48, 48, 1))
        if int(data[i][0]) == 1:
            data[i][0] == 0
        if int(data[i][0]) > 0:
            data[i][0] = int(data[i][0]) - 1
        labels[i] = np.array(data[i][0]).astype('float32')
        labels_binarized[i, int(labels[i])] = 1
    return images, labels_binarized

def save_images(train_images, train_labels, path, division = 20):
    division = 20
    size = len(train_images)
    batch_size = size // division
    categories = ['Anger', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    it_imag = 0
    it_categ = np.zeros(shape = (6,1))
    for i in range(division): 
        images = tf.image.resize(images=train_images[i*batch_size:min((i+1)*batch_size,size-1),:,:,:], 
                                 size=(224, 224), method='bilinear').numpy()
        rgb_images = np.zeros(shape=(min((i+1)*batch_size,size-1)-i*batch_size, 
                                    224, 224, 3), dtype='float32')
        for l in range(min((i+1)*batch_size,size-1)-i*batch_size):
            rgb_images[l, :, :, :] = tf.image.grayscale_to_rgb(tf.convert_to_tensor(images[l, :, :, :])).numpy()
        rgb_images_trunc = np.floor(rgb_images)
        for k in range(len(categories)):
                a = rgb_images_trunc[train_labels[i*batch_size:min((i+1)*batch_size,
                                                                     size-1), k] == 1, :, :, :]
                for m in range(a.shape[0]):
                    b = a[m, :, :, :].astype('uint8')
                    im = Image.fromarray(b, mode='RGB')
                    
                    if it_categ[m][0] < 3996 :
                        im.save(os.path.sep.join([path, "Data_Images/Training/"]) +categories[k]+"/Images"+str(it_imag)+".jpeg", 
                                    subsampling=0, quality=100)   
                        it_imag += 1
                        it_categ[m] += 1
                    
def csv_to_rgb_list(path):
    images, labels = csv_to_list(path)
    images = tf.image.resize(images=images, size=(224, 224), method='bilinear').numpy()
    rgb_images = np.zeros(shape=(images.shape[0], 224, 224, 3), dtype='float32')
    for i in range(images.shape[0]):
        rgb_images[i, :, :, :] = tf.image.grayscale_to_rgb(tf.convert_to_tensor(images[i, :, :, :])).numpy()
    return rgb_images, labels

class print_test_accuracy(tf.keras.callbacks.Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.reports = []

    def on_epoch_end(self, epoch, logs={}):
        report = tf.keras.metrics.CategoricalAccuracy()(self.y, self.model.predict(self.x)).numpy()
        self.reports.append(report)
        print("Test Accuracy", report)
        print("")
        return