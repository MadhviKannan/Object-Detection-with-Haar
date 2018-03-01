# Object-Detection-with-Haar </br>
This code runs on Python 2.7. It was tested on Mac OSX. It can be run on Linux/Mac OS.

# Modules used </br>
1) OpenCV version 3.3.0 (doesn't work for versions older than this) </br>
2) SubProcess </br>
3) Shlex </br>
4) OS </br>
5) numpy </br>
6) pillow </br>

(Recommended to install all packages in a conda virtual environment) </br>

# Training the dataset </br>
This project is an attempt to train a Cascade Classifier using Haar Transform for phone object detection. </br>
Make sure to have the find_phone directory with the 132 positive phone images along with the corresponding labels.txt </br>
The model has been trained for 10 stages and a cascade.xml file has been created in the ~/data directory. </br>

# To run this code </br>

For training : </br>
python train_phone_finder.py {arg1} </br>
arg1- path to the directory containing all phone images along with the labels.txt </br>

# To test this code </br>

python find_phone.py {arg1} </br>
{arg1} - path to the test .jpg image

# Sample Output for test image </br>


Input - ~/find_phone/89.jpg </br>
Predicted Output central coordinates- 0.26734 0.36809 </br>


# Possible next steps </br>

Increasing the dataset would be the first step towards getting a more accurate model. Currently, the negative images/background images are being retrieved from a website online (http://haar.thiagohersan.com/haartraining/negatives/) which has all images except for images with a phone. The model's accuracy might increase if the negative images just consisted of background positive image without the phone in them. </br>

State of the art for object detection is to use Neural networks - MobileNet, SSD networks. </br>
This could be a possible next step towards achieving a better, more accurate model. </br>




