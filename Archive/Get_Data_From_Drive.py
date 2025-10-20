#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 11:43:04 2022

@author: ionc4
"""


import argparse
import os
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required = True,
                 help = "path to project folder")
args = vars(ap.parse_args())



if not os.path.isdir(os.path.sep.join([args["path"],'Data'])):
    import requests

    os.mkdir(os.path.sep.join([args["path"],'Data']))

    url = 'https://www.dropbox.com/s/gqidge0hlojkexa/data_train.csv?dl=1'
    r = requests.get(url, allow_redirects=True)
    open(os.path.sep.join([args["path"],'Data/data_train.csv']), 'wb').write(r.content)

    url = 'https://www.dropbox.com/s/7j502y4lc4juyem/data_val.csv?dl=1'
    r = requests.get(url, allow_redirects=True)
    open(os.path.sep.join([args["path"],'Data/data_val.csv']), 'wb').write(r.content)

    url = 'https://www.dropbox.com/s/92o63vgr6rler2a/data_test.csv?dl=1'
    r = requests.get(url, allow_redirects=True)
    open(os.path.sep.join([args["path"],'Data/data_test.csv']), 'wb').write(r.content)