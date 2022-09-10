# -*- coding: utf-8 -*-

"""
Created on Sat Aug  6 08:47:16 PM 2022

@author: ion.ceparu
"""

from keras.utils import np_utils
import numpy as np
import h5py
import cv2


class HDF5_To_Data: 
    def __init__(self, db_path, dim_batch, preprocesor = None, augm = None, 
                 binarize = True, n_labels = 2):
        #db_path -> path to dataset
        #dim_batch -> size of batches to supply while training
        #preprocesor -> list that contains the preproccesing of the images
        #augm -> apply augmentation or not
        #binarize -> binarize the labels
        #labels -> number of labels in the dataset
        self.db = h5py.File(db_path)
        self.n_imag = self.db["labels"].shape[0]

        self.dim_batch = dim_batch
        self.preprocesor = preprocesor
        self.augm = augm
        self.binarize = binarize
        self.n_labels = n_labels
        
    def to_data(self, max_n_ep = np.inf):
        
        #epoch counter
        n_ep = 0
        
        while n_ep < max_n_ep :
            #for each batch in the dataset 
            for i in np.arange(0, self.n_imag, self.dim_batch):
                #extract the images and labels
                data = self.db["images"][i : i + self.dim_batch]
                labels = self.db["labels"][i : i + self.dim_batch]
                #check if we binarize the labels
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.n_labels)
                #check to see if we have the proccesing steps
                if self.preprocesor is not None:
                    #use a list for the proccesed images
                    procesed_images = []
                    for image in data:                   
                        for pp in self.preprocesor:
                            image = pp.preprocesare(image)
                        procesed_images.append(image)
                    #update the data
                    data = np.array(procesed_images)
                    
                #check if we need to apply data augmentation     
                if self.augm is not None:
                    (data, labels) = next(self.augm.flow(data, labels,
                                                          batch_size = self.dim_batch))

                yield (data, labels)
            #increment the no_of_epochs
            n_ep += 1
    
    def close(self):
        self.db.close()
        