# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 13:01:45 2022

@author: ion.ceparu
"""

import argparse
import os
ap = argparse.ArgumentParser()

ap.add_argument("-dp", "--datapath", required = True,
                 help = "path to Data folder")
ap.add_argument("-pp", "--projectpath", required = True,
                 help = "path to project folder")
ap.add_argument("-op", "--outputpath", required = True,
                 help = "path to output folder")
args = vars(ap.parse_args())

from Scripts.Useful_Scripts import csv_to_list
from Scripts.Useful_Scripts import save_images
#exec(open(os.path.sep.join([args["projectpath"], "Scripts/Useful_Scripts.py"])).read())


# if not os.path.isdir(os.path.sep.join([args["outputpath"],'Data_Images'])):
from PIL import Image



    # os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images']))
    # os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training']))
    # os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Anger']))
    # os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Disgust']))
    # os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Fear']))
    # os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Happiness']))
    # os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Sadness']))
    # os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Surprise']))
    # os.mkdir(os.path.sep.join([args["outputpath"],'Data_Images/Training/Neutral']))

train_images, train_labels = csv_to_list(os.path.sep.join([args["datapath"],"data_test.csv"]))
    
save_images(train_images, train_labels, path = args["outputpath"])
   
    

