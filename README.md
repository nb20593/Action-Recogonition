### Introduction
The computer vision community is actively researching the ability to recognise human action from video. In this project, Python is used to create a prototype model for recognising human action from video

### Built With
Python

### Prerequsites
Install anaconda or install the necessary packages from spec-file.txt 

or 

conda create --name <env_name> --file spec-file.txt

### Dataset
KTH dataset comprises the videos of human actions boxing, handclapping, hand waving, jogging, running, and walking performed by six different people. These acts are carried out by 25 individuals in 4 settings: inside, outdoor with scale variation, outdoor with various clothing, and outdoor. Therefore, 25x4x6 = 600 videos make up the entire collection. The videos have a frame rate of 25 frames per second and a 160x120 resolution. On the website, you may check for more details about the dataset.[https://www.csc.kth.se/cvap/actions/]

Create a folder named dataset and download all the action categories along with the sequence.txt file

### 3D CNN + optical flow
Optical flow is a technique used to describe image motion. It is usually applied to a series of images that have a small time step between them, for example, video frames. Optical flow calculates a velocity for points within the images, and provides an estimation of where points could be in the next image sequence.

There are different types of algorithm for optical flow Dense Pyramid Lucas-Kanade, Farneback, PCAFlow, SimpleFlow, RLOF, DeepFlow, DualTVL1. In this project we are using Farneback

### Architecture
![3d cnn + opticalflow.png](https://github.com/nb20593/Action-Recogonition/blob/main/3d%20cnn%20%2B%20opticalflow.png)

### Training and Evaluation
Change Batch_size, number of epoch and start epoch , Monitor the accuracy in every epoch. To validate the data with the trained model select a model from model checkpoint and replace epoch40 in the code and update the model name 

![Train Accuracy.png](https://github.com/nb20593/Action-Recogonition/blob/main/Train%20Accuracy.png)

### Results
The accuracy of this model on the Validation set is 92.12%
