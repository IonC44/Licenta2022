# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 13:01:45 2022

@author: ion.ceparu
"""

import argparse
import os
from PIL import Image
ap = argparse.ArgumentParser()

ap.add_argument("-dp", "--datapath", required = True,
                 help = "path to Data folder")
ap.add_argument("-pp", "--projectpath", required = True,
                 help = "path to project folder")
ap.add_argument("-op", "--outputpath", required = True,
                 help = "path to output folder")
args = vars(ap.parse_args())

from Scripts.Useful_Scripts import *
#exec(open(os.path.sep.join([args["projectpath"], "Scripts/Useful_Scripts.py"])).read())

categories = ['Anger', 'Fear','Disgust', 'Happiness', 'Sadness', 'Surprise', 'Neutral','Unknown']

if not os.path.isdir(os.path.sep.join([args["outputpath"],'Data_Images'])):




    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Anger']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Disgust']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Fear']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Happiness']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Sadness']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Surprise']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Neutral']))
    os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Unkown']))

train_images, train_labels = csv_to_list_fer_plus(os.path.sep.join([args["datapath"],"data_train.csv"]), 
                                                  label_path = "/home/ionc4/FERPlus/data/FER2013Train/label.csv",
                                                  num_categ = len(categories))
    
save_images(train_images, train_labels,categories=categories,scope = "Training", path = args["outputpath"])
   
    

