#This script wil translate the dataset data intro the HDF5 format(Hierarchical Data Format5)

"""
Created on Sat Aug  6 10:42:42 2022

@author: ion.ceparu
"""

import h5py
import os

class data_to_hdf5:
    def __init__(self, dim, output_path, data_type = "images", dim_buff = 1024):
        #Check the output_path 
        if os.path.exists(output_path):
            print("[ERROR] : Output file: {} still exists".format(output_path))
            os.remove(output_path)
            print("[ERROR] : Output file: {} deleted".format(output_path))
            
        #create a hdf5 database which has 2 datasets
        #1 -> features
        #2 -> labels
        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_type, dim, dtype = "float")
        self.labels = self.db.create_dataset("labels", (dim[0], ), dtype = "int")
        
        #memorize the size of the buffer 
        self.dim_buff = dim_buff
        self.buffer = {"data": [],"labels" : []}
        self.data_start = 0
        
    def add_data(self, new, label):
        #adds new rows and labels to the buffer created earlyer
        self.buffer["data"].extend(new)
        self.buffer["labels"].extend(label)

        #If the buffer is full
        if len(self.buffer["data"]) >= self.dim_buff:
            self.write_reset()

    def write_reset(self):
        #write the content of the buffer then delete it
        i = self.data_start + len(self.buffer["data"])
        self.data[self.data_start:i] = self.buffer["data"]
        self.labels[self.data_start:i] = self.buffer["labels"]
        self.data_start = i
        self.buffer = {"data": [], "labels": []}

    def new_label_dataset(self, class_labels):
        dataset = self.db.create_dataset("label_text", (len(class_labels,)))
        dataset[:] = class_labels

    def close(self):
        #check if the there is still data in the buffer
        if len(self.buffer["data"]) > 0:
            self.write_reset()

        self.db.close()
