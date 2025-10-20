# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 13:01:45 2022

@author: ion.ceparu
"""

import argparse
import os
from PIL import Image
from Scripts.Useful_Scripts import *

# Add arguments for the script
ap = argparse.ArgumentParser()

ap.add_argument("-dp", "--datapath", required = True,
                 help = "path to Data folder")
ap.add_argument("-pp", "--projectpath", required = True,
                 help = "path to project folder")
ap.add_argument("-op", "--outputpath", required = True,
                 help = "path to output folder")
args = vars(ap.parse_args())

categories = ['Anger', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

if not os.path.isdir(os.path.sep.join([args["outputpath"],'Data_Images'])):

    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Anger']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Fear']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Happiness']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Sadness']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Surprise']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Neutral']))
    
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Validation']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Validation/Anger']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Validation/Fear']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Validation/Happiness']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Validation/Sadness']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Validation/Surprise']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Validation/Neutral']))

    train_images, train_labels = csv_to_list(os.path.sep.join([args["datapath"],"data_train.csv"]), 
                                                  num_categ = len(categories))
    save_images(train_images, train_labels,categories=categories,scope = "Training", path = args["outputpath"])
   
    train_images, train_labels = csv_to_list(os.path.sep.join([args["datapath"],"data_val.csv"]), 
                                                  num_categ = len(categories))
    save_images(train_images, train_labels,categories=categories,scope = "Validation", path = args["outputpath"])

