# -*- coding: utf-8 -*-

"""
Created on Sat Aug  6 07:43:59 PM 2022

@author: ion.ceparu
"""

from keras.callbacks import Callback
import os

#We now inherit from Callback class
class EpochCallback(Callback):
    
    def __init__(self, output_path, each = 1, start_at = 0):
        
        #call the Callback __init__
        super(Callback, self).__init__()
        
        #store the output_path of the model
        #number of epoch after which the model will be written to the disk
        #the current value of the epoch
        
        self.output_path = output_path
        self.each = each
        self.intEpoch = start_at
        
    def on_epoch_end(self, epoch, logs = {}):
        #check to see if the model should be written to the disk
        if (self.intEpoch + 1) % self.each == 0:
             self.model.save(os.path.sep.join([self.output_path, 
                                               "epoch#{}.hdf5".format(self.intEpoch + 1)]),
                             overwrite = True)
        #increment at each epoch
        self.intEpoch += 1
        
        
        
    