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
train_df = dt.train_df
test_df = dt.test_df

data_directory = os.path.join(os.getcwd(), "skin_data")
batch_size = 32
img_height = 180
img_width = 180
target_size = (180,180)

# Data generator

# Specify Image Data Generator 

datagen=ImageDataGenerator(horizontal_flip= True,
                            shear_range= 0.2, # Shear angle in counter-clockwise direction in degrees
                            zoom_range=0.2, #Range for random zoom
                            rotation_range=20, #Degree range for random rotations.
                            rescale=1./255.) # rescaling factor 



train_ds = datagen.flow_from_dataframe(
            dataframe= train_df,
            #directory= df["filepath"] #none because the filepath is complete
            x_col="filepaths",
            y_col="labels",
            batch_size=batch_size,
            seed=666,
            shuffle=True,
            class_mode= "binary",
            validation_split=0.2,
            target_size=target_size)


