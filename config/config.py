# -*- coding: utf-8 -*-

#DATABASE PATH
DATABASE_PATH ="Database/ker2013/ker2013/ker2013.csv"

#Paths for HDF5 FILES(Training, Validation and Testing Data)

TRAIN_DATA_HDF5 = "hdf5/train.hdf5"
VALID_DATA_HDF5 = "hdf5/valid.hdf5"
TEST_DATA_HDF5 = "hdf5/test.hdf5"

#batch dimension
DIM_BATCH = 32
#Path of the output 
OUTPUT_PATH ="output"

#Number of labels for identification: (0=Angry, 1=Disgust, 2=Fear, 3=Sad, 4=Neutral)
N_LABELS = 5