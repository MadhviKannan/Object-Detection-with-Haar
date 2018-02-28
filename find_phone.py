#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:25:28 2018

@author: madhvikannan
"""

import numpy as np
import cv2
import sys
import os
testDir=sys.argv[1]

dataDir=os.getcwd()

watch_cascade = cv2.CascadeClassifier(dataDir+'/data/cascade.xml')


img = cv2.imread(testDir)

width=490
height=326

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
watches = watch_cascade.detectMultiScale(gray, 25, 25)
    
    # add this
for (x,y,w,h) in watches:
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    unnormalized_x=x+25
    unnormalized_y=y+25
    x_new=float(unnormalized_x)/float(width)
    y_new=float(unnormalized_y)/float(height)

print x_new, y_new
