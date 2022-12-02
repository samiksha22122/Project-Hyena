# Hyena detection on Yolov5 using Jetson Nano 2gb
# Aim And Objectives
# Aim :-
To create a Hyena detection system which will detect hyena's body mark. Then it will classify which species it belongs.

# Objective :-
• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module whether a hyena belongs to which species.

# Abstract :-
• A hyena is classified whether a hyena is belonging to which species and is detected by the live feed from the system’s camera.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

• The purpose of this project is to observe hyena behaviour.


# Introduction :-
• This project is based on a Hyena detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.

• This project can also be used to gather information about which species does the hyena belong.

• Hyena can be classified into brown hyena,spotted hyena and striped hyena based on the image annotation we give in roboflow.

• Hyena detection sometimes become difficult as face or body are covered up by grass, trees and rock thereby making hyena detection difficult. However, training in Roboflow has allowed us to crop images and change the contrast of certain images to match the time of day for better recognition by the model.

• Neural networks and machine learning have been used for these tasks and have obtained good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for hyena detection as well.

# Jetson Nano Compatibility :-
The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

# Jetson Nano 2gb :-
![202396722-2296241b-c181-4530-b199-d4fda877f8c3](https://user-images.githubusercontent.com/117035158/205257423-da9e91e4-330a-49d0-a740-b244aad1a8bd.jpg)



# Proposed System :-
1] Study basics of machine learning and image recognition.

2]Start with implementation

• Front-end development

• Back-end development

3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether the hyena belongs to which species.

4] Use datasets to interpret the object and suggest whether the hyena on the camera’s viewfinder belongs to which species.

# Methodology :-
The hyena detection system is a program that focuses on implementing real time Hyena detection. It is a prototype of a new product that comprises of the main module: Hyena detection and then showing on viewfinder whether the hyena belongs to brown,spotted or striped Species.

This Module is described as follows +:

• Ability to detect the location of a hyena's face in any input image or frame. The output is the bounding box coordinates on the detected body and face of hyena.

• For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from google images and made our own dataset.

• This Datasets identifies hyena's face and body in a Bitmap graphic object and returns the bounding box image with annotation of Brown hyena, Spotted hyena or Striped hyena present in each image.

• Recognition of the face and body and whether the hyena belongs to which species.

• Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

• YOLOv5 was used to train and test our model for whether the hyena belongs to which species. We trained it for 149 epochs and achieved an accuracy of approximately 92%.

# Installation :-
# Initial Configuration :-

sudo apt-get remove --purge libreoffice*

sudo apt-get remove --purge thunderbird*

Create Swap :-

udo fallocate -l 10.0G /swapfile1

sudo chmod 600 /swapfile1

sudo mkswap /swapfile1

sudo vim /etc/fstab

#make entry in fstab file

/swapfile1 swap swap defaults 0 0

Cuda env in bashrc :-

vim ~/.bashrc

#add this lines

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

Update & Upgrade :-

sudo apt-get update

sudo apt-get upgrade

Install some required Packages :-

sudo apt install curl

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

sudo python3 get-pip.py

sudo apt-get install libopenblas-base libopenmpi-dev

sudo pip3 install pillow

Install Torch :-

curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl

mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True"

sudo python3 -c "import torch; print(torch.cuda.is_available())"

Install Torchvision :-

git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision

cd torchvision/

sudo python3 setup.py install

Clone Yolov5 :-

git clone https://github.com/ultralytics/yolov5.git

cd yolov5/

sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1

sudo pip3 install -r requirements.txt

Download weights and Test Yolov5 Installation on USB webcam :-

sudo python3 detect.py

sudo python3 detect.py --weights yolov5s.pt --source 0

# Hyena Dataset Training :-
# We used Google Colab And Roboflow :-
train your model on colab and download the weights and past them into yolov5 folder link of project colab file given in repo

# Running Hyena Detection Model :-
source '0' for webcam

!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0

# Demo :-
link- https://youtu.be/4jzewscEWk4

# Advantages :-
• Hyena detection system will be of great help in locating hyena in jungle safari.

• Hyena detection system shows whether the hyena in viewfinder of camera module is brown hyena, spotted hyena or striped hyena with good accuracy.

• It can then convey it to authorities like forest officer or the data about the respective hyena where he is relocatting and it can help the forest department to spot the hyena easily.

• When completely automated no user input is required and therefore works with absolute efficiency and speed.

• It can work around the clock and therefore becomes more cost efficient.

# Application :-
• Detects a hyena’s face and body and then checks whether the hyena belongs to which species in each image frame or viewfinder using a camera module.

• Can be used anywhere in forest as the hyena usually roam and hyena detection becomes even more accurate.

• Can be used as a reference for other ai models based on Hyena Detection.

# Future Scope :-
• As we know technology is marching towards automation, so this project is one of the step towards automation.

• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

• Hyena detection will become a necessity in the future due to decrease in population and hence our model will be of great help to locate the hyena in an efficient way.

# Conclusion :-
• In this project our model is trying to detect a hyena’s face or body and then showing it on viewfinder, live as whether hyena belongs to which species as we have specified in Roboflow.

• The model tries to solve the problem of severe injuries and attack of hyena to human that occure in forest and thus protects a person’s life.

• The model is efficient and highly accurate and hence reduces the workforce required.

# Reference :-
1] Roboflow:- https://roboflow.com/

2] Google images
