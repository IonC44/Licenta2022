
"""
Created on Sat Aug  6 07:58:59 PM 2022

@author: ion.ceparu
"""

from keras.callbacks import History
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime

class TrainMonitor(History):
    def __init__(self, save_path, start_at = 0):
        #parent __init__
        super().__init__()
        #output path for the statistics figure
        #path to json
        #starting epoch
        self.save_path = save_path
        self.start_at = start_at

        # Check that directory exists
        os.makedirs(self.save_path, exist_ok=True)
    
    def on_train_begin(self, logs = {}):
        
        #initalize the history
        self.his = {}
        
        
    def on_epoch_end(self, epoch, logs = {}):
        #iterate the log and update
        for (k, v) in logs.items():
            aux = self.his.get(k, [])
            aux.append(v)
            self.his[k] = aux
        
        #if at least two epochs have passed 
        if len(self.his["loss"]) > 1:

            #plot the loss and accuracy
            N = np.arange(0, len(self.his["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.his["loss"], label="training_loss")
            plt.plot(N, self.his["val_loss"], label="validation_loss")
            plt.plot(N, self.his["SparseCategoricalCrossentropy"], label="training_sparse_categorical_crossentropy")
            plt.plot(N, self.his["val_SparseCategoricalCrossentropy"], label="valid_sparse_categorical_crossentropy")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
				len(self.his["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            
            #save figure
            now = datetime.datetime.now().strftime('%y %m %d %H')

            # Save JSON
            file = open(os.path.sep.join([self.save_path, 'JSON_' + now + '.json']), "w")
            file.write(json.dumps(self.his))
            file.close()
            
            plt.savefig(os.path.sep.join([self.save_path, 'PLOT_' + now + '.png' ]))
            plt.close()
            