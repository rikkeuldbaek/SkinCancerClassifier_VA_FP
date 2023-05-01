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
sys.path.append(os.path.join("utils"))
import helper_func as hf

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
import time


#load in data
train_df = dt.train_df
test_df = dt.test_df


#################### Prepping variables ####################

#data_directory = os.path.join(os.getcwd(), "skin_data")
batch_size = 32
img_height = 180
img_width = 180
target_size = (180,180)
n_epochs = 20

#################### Data generator ####################

# Specify Image Data Generator 

datagen=ImageDataGenerator(horizontal_flip= True,
                            shear_range= 0.2, # Shear angle in counter-clockwise direction in degrees
                            zoom_range=0.2, #Range for random zoom
                            rotation_range=20, #Degree range for random rotations.
                            rescale=1./255.,# rescaling factor 
                            validation_split=0.2) # validation split


# training data
train_ds = datagen.flow_from_dataframe(
                    dataframe= train_df,
                    #directory= df["filepath"] #none because the filepath is complete
                    x_col="filepaths",
                    y_col="labels",
                    batch_size=batch_size,
                    seed=666,
                    shuffle=True,
                    class_mode= "binary",
                    subset="training",
                    target_size=target_size)


# validation data
val_ds =datagen.flow_from_dataframe(
                    dataframe=train_df,
                    #directory= df["filepath"] #none because the filepath is complete
                    x_col="filepaths",
                    y_col="labels",
                    batch_size=batch_size,
                    seed=666,
                    subset="validation",
                    shuffle=True,
                    class_mode= "binary",
                    target_size=target_size)


test_datagen=ImageDataGenerator(rescale=1./255.)

# test data
test_ds =test_datagen.flow_from_dataframe(
                    dataframe=test_df,
                    #directory= df["filepath"] #none because the filepath is complete
                    x_col="filepaths",
                    y_col="labels",
                    batch_size=batch_size,
                    seed=666,
                    shuffle=False,
                    class_mode= "binary",
                    target_size=target_size)


# output from generators
print(train_ds.class_indices.keys())#unique labels ['Cancer', 'Non_cancer']
print(train_ds.class_indices)#unique labels ['Cancer':0, 'Non_cancer':1 ]


############## PLOT SAMPLE ################




############## CHECK DATA ################
#check shapes of generated train_ds
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    #check the images are standardized 
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image)) # pixel values should be between 0 and 1
    break



############## MODEL ################
input_shape = (180,180,3)

model = tf.keras.models.Sequential([
    # since Conv2D is the first layer of the neural network, we should also specify the size of the input
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
    # apply pooling
    tf.keras.layers.MaxPooling2D(2,2),
    # and repeat the process
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # flatten the result to feed it to the dense layer
    tf.keras.layers.Flatten(), 
    # and define 512 neurons for processing the output coming by the previous layers
    tf.keras.layers.Dense(512, activation='relu'), 
    # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
model.summary()


############## FIT & TRAIN #################
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

epochs = 20
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs)


loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)




