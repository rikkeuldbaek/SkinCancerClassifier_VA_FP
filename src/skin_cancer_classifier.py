### Skin Cancer Detection Model
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 27th of April 2023

#--------------------------------------------------------#
################ SKIN CANCER CLASSIFIER ##################
#--------------------------------------------------------#

# Install packages
# data wrangeling/path tools/plotting tools 
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt

# data 
import data as dt

# tf tools 
import tensorflow as tf
 
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10 

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# call backs
from tensorflow.keras.callbacks import EarlyStopping

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report


# Import predefined helper functions (for plotting)
#sys.path.append(os.path.join("utils"))
#import helper_func as hf

#load in data
df = dt.df

# Data generator

# Specify Image Data Generator

datagen=ImageDataGenerator(horizontal_flip= True,
                            shear_range= 0.2, # Shear angle in counter-clockwise direction in degrees
                            zoom_range=0.2, #Range for random zoom
                            rotation_range=20, #Degree range for random rotations.
                            rescale=1./255.) # rescaling factor 


# Train generator
train_generator=datagen.flow_from_dataframe(
    dataframe=train_df,
    directory= os.path.join("..","..", ".."), #my path
    x_col="image_path",
    y_col="class_label",
    batch_size=batch_size,
    seed=666,
    shuffle=True,
    class_mode= class_mode,
    target_size=target_size)

# Validation generator
val_generator=datagen.flow_from_dataframe(
    dataframe=val_df,
    directory= os.path.join("..","..", ".."), #my path
    x_col="image_path",
    y_col="class_label",
    batch_size=batch_size,
    seed=666,
    shuffle=True,
    class_mode= class_mode,
    target_size=target_size)

    # Validation generator
test_generator=datagen.flow_from_dataframe(
    dataframe=test_df,
    directory= os.path.join("..","..", ".."), #my path
    x_col="image_path",
    y_col="class_label",
    batch_size=batch_size,
    seed=666,
    shuffle=False,
    class_mode= class_mode,
    target_size=target_size)