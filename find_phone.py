# This code gives the predictions for any test .jpg image based on the cascade.xml file generated. 

import numpy as np
import cv2
import sys
import os
testDir=sys.argv[1] #path to test image

dataDir=os.getcwd()

watch_cascade = cv2.CascadeClassifier(dataDir+'/data/cascade.xml') #object to store cascade.xml file


img = cv2.imread(testDir)

width=490
height=326

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
watches = watch_cascade.detectMultiScale(gray, 25, 25)
    
for (x,y,w,h) in watches:
    unnormalized_x=x+25 
    unnormalized_y=y+25
    x_new=float(unnormalized_x)/float(width) #getting normalized x,y coordinates
    y_new=float(unnormalized_y)/float(height)

print x_new, y_new
