#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:53:54 2018

@author: madhvikannan
"""
#Modules required

import cv2
from PIL import Image
import numpy as np
import os
import subprocess
import shlex
import sys
rootDir=sys.argv[1]

#rootDir='/Users/madhvikannan/Desktop/USC_Docs/BrainCorp/find_phone_task/find_phone'
dataDir=os.getcwd()
dataDir=dataDir+'/data'
print dataDir

if not os.path.exists(dataDir):
    os.mkdir(dataDir)
else:
    print 'Data directory ready for training'
#destDir='/Users/madhvikannan/Desktop/USC_Docs/BrainCorp/find_phone_task/find_positive_phone'
rootDirNeg=rootDir+'/'+'NegativePhone'
destDirNeg=rootDir+'/'+'NegativePhoneResized'

fileList=os.listdir(rootDir)
imageArray=[]
#Converting images to numpy arrays of shape (num_images, height, width, num_channels)
for fileName in fileList:
    if fileName.endswith('.jpg'):
        
        im=Image.open(rootDir+'/'+fileName)
        im2arr=np.array(im)
        imageArray.append(im2arr)
imageArray=np.array(imageArray)

height=imageArray.shape[1]
width=imageArray.shape[2]
print height,width
print imageArray.shape

#%%
#Converting labels.txt to format label_name, num_class, x_coord, y_coord, box_height, box_width
box_length=50
box_breadth=50
f=open(rootDir+'/'+'labels.txt','r')
lines=f.readlines()
new_x=[]
new_y=[]
bottom_x=[]
bottom_y=[]
k = 0
with open(rootDir+'/'+'positive.lst','a') as f:
    for x in lines:
        x=x.strip()
        old_height=float(x.split(' ')[1])
        old_width=float(x.split(' ')[2])
        new_y.append(height*float(old_width) - 25)
        new_x.append(width*float(old_height)- 25)
        bottom_x.append(width*float(old_height)+25)
        bottom_y.append(height*float(old_width) + 25)
        f.write(str(x.split(' ')[0])+" 1 "+str(int(width*(old_height)-25)) +" "+str(int(height*(old_width)-25))+" 25 25"+"\n")
f.close()

## Get same number of Negative Images and resize to same size as positive images
if not os.path.exists(destDirNeg):
    os.mkdir(destDirNeg) 
    
if not os.path.exists(rootDirNeg):
    os.mkdir(rootDirNeg)
else:
    print 'directory exists'
    
os.chdir(rootDirNeg)

commandline = 'wget -nd -r -A "neg-0[0-3]*.jpg" http://haar.thiagohersan.com/haartraining/negatives/'
args=shlex.split(commandline)
print args
if os.listdir(rootDirNeg)==[]:
    p=subprocess.call(args)
    print 'done'
else:
    print 'files present'
  
#Get 130 images from the negative image directory and resize it to the same size as the positive images

    
file_name='neg.txt'
fileNames=os.listdir(rootDirNeg)
with open(destDirNeg+'/'+file_name,'w') as f:
                for i in range(129):
                    if fileNames[i].endswith('.jpg'):
                        image=cv2.imread(rootDirNeg+'/'+fileNames[i],cv2.IMREAD_GRAYSCALE)
                        image_resized=cv2.resize(image,(326,490))
                        cv2.imwrite(destDirNeg+'/'+str(fileNames[i]),image_resized)
                        
                        f.write(rootDirNeg+"/"+fileNames[i]+'\n')
f.close()

commandline1='opencv_createsamples -info /Users/madhvikannan/Desktop/USC_Docs/BrainCorp/find_phone_task/find_phone/positive.lst -num 1950 -w 25 -h 25 -vec /Users/madhvikannan/Desktop/USC_Docs/BrainCorp/find_phone_task/find_phone/positives.vec'
args1=shlex.split(commandline1)
print 'creating positive vector file'
p=subprocess.call(args1)
print 'done'

commandline2='opencv_traincascade -data ' + dataDir + ' -vec ' + rootDir + '/positives.vec' +' -bg '+ destDirNeg +'/neg.txt'+' -numPos 133 -numNeg 142 -numStages 10 -w 25 -h 25'
args2=shlex.split(commandline2)
print args2
print 'starting training'
p=subprocess.call(args2)
print 'done training'



