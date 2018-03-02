#This code is used to train the Cascade classifier. It outputs a cascade.xml file in the data directory which has the trained 
#model. 
#Modules required

import cv2
from PIL import Image
import numpy as np
import os
import subprocess
import shlex
import sys
rootDir=sys.argv[1] #path to the folder containing the positive images and labels.txt

# Prepare data directory for placing trained .xml files
dataDir=os.getcwd()
dataDir=dataDir+'/data'
if not os.path.exists(dataDir):
    os.mkdir(dataDir)
else:
    print 'Data directory ready for training'

    
# Prepare the Negative directory to download and place the resized gray scale background images
rootDirNeg=rootDir+'/'+'NegativePhone' #folder to download the negative images from the web
destDirNeg=rootDir+'/'+'NegativePhoneResized' #folder to place the resized negative images

if not os.path.exists(destDirNeg):
    os.mkdir(destDirNeg) 
    
if not os.path.exists(rootDirNeg):
    os.mkdir(rootDirNeg)

    
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

#Converting labels.txt to format label_name, num_class, x_coord, y_coord, box_height, box_width

f=open(rootDir+'/'+'labels.txt','r')
lines=f.readlines()
with open(rootDir+'/'+'positive.lst','a') as f:
    for x in lines:
        x=x.strip()
        old_height=float(x.split(' ')[1])
        old_width=float(x.split(' ')[2])
        f.write(str(x.split(' ')[0])+" 1 "+str(int(width*(old_height)-25)) +" "+str(int(height*(old_width)-25))+" 25 25"+"\n")
f.close()

## Get same number of Negative Images and resize to same size as positive images    
os.chdir(rootDirNeg)

commandline = 'wget -nd -r -A "neg-0[0-3]*.jpg" http://haar.thiagohersan.com/haartraining/negatives/'
args=shlex.split(commandline)

if os.listdir(rootDirNeg)==[]:
    p=subprocess.call(args)
else:
    print 'Negative directory created and has the background images'
  
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


commandline1='opencv_createsamples -info '+rootDir+ '/positive.lst -num 350 -w 25 -h 25 -vec '+rootDir+'/positives.vec'
print commandline1
args1=shlex.split(commandline1)
p=subprocess.call(args1)
print 'Done creating vector file'

commandline2='opencv_traincascade -data ' + dataDir + ' -vec ' + rootDir + '/positives.vec' +' -bg '+ destDirNeg +'/neg.txt'+' -numPos 133 -numNeg 142 -numStages 10 -w 25 -h 25'
args2=shlex.split(commandline2)
print args2
print 'Start Training'
p=subprocess.call(args2)
print 'Done Training'



