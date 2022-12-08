# face_emotion_recognition

This simple program uses convolutional neuro networks for face detection and emotion recognition.
Face emotion recognition by computer vision is a many times solved problem. But it will become
a basement for more complicated task - attention monitoring.

# short_description

IMPORTS

numpy
mtcnn
tensorflow.keras
opencv
flask
matplotlib.pyplot as plt
numpy
os
shutil

DATASET

I've used faces dataset from Kaggle (https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer).
The dataset contain 35,685 examples of 48x48 pixel gray scale images of faces divided into train and test dataset.
Images are categorized based on the emotion shown in the facial expressions (happiness, neutral, sadness, anger, surprise, disgust, fear).
I had to review all the images, when found lots of duplicates and wrong emotions in folders.
Duplicates removal code is in duplicates_delete.ipynb file.



