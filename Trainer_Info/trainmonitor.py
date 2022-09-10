
"""
Created on Sat Aug  6 07:58:59 PM 2022

@author: ion.ceparu
"""

from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainMonitor(BaseLogger):
    def __init__(self, fig_path, json_path = None, start_at = 0):
        #parent __init__
        super().__init__()
        #output path for the statistics figure
        #path to json
        #starting epoch
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at
    
    def on_train_begin(self, logs = {}):
        
        #initalize the history
        self.his = {}
        
        #if the json path exists, load the training his
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.his = json.loads(open(self.json_path).read())
        
                if self.start_at > 0:
                    #remove the entries that are past the starting epoch
                    for k in self.his.keys():
                        self.his[k] = self.his[k][:self.start_at]
        
    def on_epoch_end(self, epoch, logs = {}):
        #iterate the log and update
        for (k, v) in logs.items():
            aux = self.his.get(k, [])
            aux.append(v)
            self.his[k] = aux
    
        #check to see if the his should be written to the file
        if self.json_path is not None:
            file = open(self.json_path, "w")
            file.write(json.dumps(self.his))
            file.close()
        
        #if at least two epochs have passed 
        if len(self.his["loss"]) > 1:
            #plot the loss and accuracy
            N = np.arange(0, len(self.his["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.his["loss"], label="training_loss")
            plt.plot(N, self.his["val_loss"], label="validation_loss")
            plt.plot(N, self.his["accuracy"], label="training_accuracy")
            plt.plot(N, self.his["val_accuracy"], label="valid_accuracy")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
				len(self.his["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            
            #save figure
            
            plt.savefig(self.fig_path)
            plt.close()
            