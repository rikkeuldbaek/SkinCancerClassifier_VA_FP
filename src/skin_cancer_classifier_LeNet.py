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

# data tools 
import os
import numpy as np
import matplotlib.pyplot as plt

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K



# Import predefined helper functions (for plotting)
sys.path.append(os.path.join("utils"))
import helper_func as hf

#load in data
train_df = dt.train_df
test_df = dt.test_df
val_df = dt.val_df

#################### Prepping variables ####################

batch_size = 32
img_height = 224
img_width = 224
target_size = (224,224)
n_epochs = 25
directory= os.path.join(os.getcwd(),"data","archive","images")

#################### Data generator ####################

# Specify Image Data Generator 

datagen=ImageDataGenerator(horizontal_flip= True,
                            vertical_flip=True,
                            zca_whitening = True,
                            shear_range= 0.2, # Shear angle in counter-clockwise direction in degrees
                            zoom_range=0.2, #Range for random zoom
                            brightness_range=(0.2, 0.8),
                            rotation_range=20, #Degree range for random rotations.
                            rescale=1./255.) #, # rescaling factor 
                            #validation_split=0.4) # validation split


# training data
train_ds = datagen.flow_from_dataframe(
                    dataframe= train_df,
                    directory = directory,
                    x_col="image",
                    y_col="label",
                    batch_size=batch_size,
                    seed=666,
                    shuffle=True,
                    class_mode= "categorical",
                    #subset="training", 
                    target_size=target_size)


# validation data
val_ds =datagen.flow_from_dataframe(
                    dataframe=val_df,
                    directory = directory,
                    x_col="image",
                    y_col="label",
                    batch_size=batch_size,
                    seed=666,
                    #subset="validation",
                    shuffle=True,
                    class_mode= "categorical",
                    target_size=target_size)


test_datagen=ImageDataGenerator(rescale=1./255.)

# test data
test_ds =test_datagen.flow_from_dataframe(
                    dataframe=test_df,
                    directory = directory,
                    x_col="image",
                    y_col="label",
                    batch_size=batch_size,
                    seed=666,
                    shuffle=False,
                    class_mode= "categorical",
                    target_size=target_size)




############## CHECK DATA ################
#check shapes of generated train_ds
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    #check the images are standardized 
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image)) # pixel values should be between 0 and 1
    break


############## Shallow net MODEL  ###############
# define model
model = Sequential()

# first set of layers CONV => RELU => MAXPOOL
model.add(Conv2D(32, # input nodes
                 (3,3), # kernel size
                 padding="same",
                 input_shape=(224,224,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2),
                       strides = (2,2)))

# second set of layers CONV => RELU => MAXPOOL
model.add(Conv2D(50, (5,5), 
                 padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2),
                       strides = (2,2)))

# FC => RELU
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

# softmax classifier
model.add(Dense(7))
model.add(Activation("softmax"))


#print(model.summary())





############## FIT & TRAIN #################
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs= n_epochs)


############ plot model #######
#hf.plot_history(history, n_epochs)

####### evaluate
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

predictions = model.predict(test_ds, # X_test
                            batch_size=batch_size)


# Make classification report
report=(classification_report(test_ds.classes, # y_test 
                                            predictions.argmax(axis=1),
                                            target_names=test_ds.class_indices.keys())) #labels

print(report)
# Define outpath for classification report
outpath_report = os.path.join(os.getcwd(), "out", "LeNet.txt")

# Save the  classification report
file = open(outpath_report, "w")
file.write(report)
file.close()

print( "Saving the skin cancer classification report in the folder ´out´")



