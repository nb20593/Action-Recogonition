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

### Results
![Accuracy on validation set.png](https://github.com/nb20593/Action-Recogonition/blob/main/Accuracy%20on%20validation%20set.png)
