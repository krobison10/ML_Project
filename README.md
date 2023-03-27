# Social Media User Profiling with Machine Learning
Code for a deep learning model I developed for my Intro to Machine Learning course.

<br>

## 📃 Table of Contents:

- [About](#-about-the-problem)
- [Technologies](#-technologies)
- [Setup](#%EF%B8%8F-setup)
- [Approach](#-approach)
- [Status](#-status)

<br>

## 📕 About The Problem

The challenge involved two classifcation tasks and five regression tasks. The classification tasks were predicting gender and age of social media users, and the regression tasks were predicting five personality traits which were openness, conscientiousness, extroversion, agreeablness, and neuroticism. The training data consisted of information for 9500 users of a popular social media platform. The sources were in three categories: images (profile pictures), text (status updates), and relational data (likes). We formed groups of three and each group member was to pick a source and only focus on their own source, each team member was solely in charge of their own performance. I chose images as my source, because I knew that I wanted to get some experience with deep learning. 

The quality of the images was okay at best. The issue was that the images were quite varied in size, as well as the subjects. The subjects were sometimes not very clear, sometimes there were none, sometimes there were multiple, sometimes they weren't human. This meant that the challenge was going to be preprocessing the data so that a convolutional neural network (CNN) could actually hope to learn on the dataset.

I would opt to focus on gender alone, I wanted to see how high of a score I could get.
 

<br>

## 💻 Technologies

<img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" alt="Python Logo" width="75" height="75"/><img src="https://github.com/devicons/devicon/blob/master/icons/tensorflow/tensorflow-original.svg" alt="TensorFlow Logo" width="75" height="75"/><img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original-wordmark.svg" alt="OpenCV Logo" width="75" height="75"/><img src="https://github.com/devicons/devicon/blob/master/icons/pandas/pandas-original-wordmark.svg" alt="Pandas Logo" width="75" height="75"/><img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" alt="NumPy Logo" width="75" height="75"/><img src="https://github.com/devicons/devicon/blob/master/icons/linux/linux-original.svg" alt="Linux Logo" width="75" height="75"/>

<br>

## 🛠️ Setup
1. The data cannot be accessed, so there is nothing to set up. This repository is read-only.

<br>

## 🤓 Approach

### Preprocessing
All the data was manipulated with pandas, I stored user IDs, gender labels, and images in a pandas dataframe. For preprocessing I used OpenCV. I looped through every image in the training data and applied the following steps.
  - Attempt to detect face with a haar cascade classifier
  - If anything but one single face detected, remove from training data
  - Crop the image to fit the detected face
  - Resize to 96 x 96
  - Convert to grayscale (better accuracy, less dimensionality maybe?)
  - Normalize pixel values to between 0 and 1
  
Using this technique, only 13% of the dataset had to be removed. What was left was a set of very consistent images. Note that these identical steps are used at classification time, except that if a single face isn't detected then, another model or the baseline prediction is used. 

### Model Training
The model was built with the Keras API for TensorFlow. This allowed me to quickly experiment with different architectures. I constructed about a dozen different architectures with likely dozens of different variations of each. The winning architecture is marked in the script `architectures.py`. 

It achieved an accuracy well into the 80% range, up to 87% on the filtered images. This was far beyond the satisfactory level of performance, and it helped secure me a spot on a team that my professor was forming for a machine learning competition. At that point I had to move on and shelf this project for good.

<br>

## 📈 Status
This project is complete. As part of the agreement to use the data, I was required to relinqish my possession of it at the conclusion of the course.

<br>
