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
import data2 as dt

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

#load in data
train_df = dt.train_df
test_df = dt.test_df
val_df = dt.val_df


#################### Prepping variables ####################

batch_size = 32
img_height = 224
img_width = 224
target_size = (224,224)
n_epochs = 2
directory= os.path.join(os.getcwd(),"data","archive","images")

#################### Data generator ####################

# Specify Image Data Generator 

datagen=ImageDataGenerator(horizontal_flip= True,
                            vertical_flip=True,
                            #zca_whitening = True,
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


# output from generators
print(train_ds.class_indices.keys())#unique labels ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
print(train_ds.class_indices)#unique labels and numerical value {'AKIEC': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'VASC': 6}


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



############## LOAD MODEL ################
# load the pretrained VGG16 model without classifier layers
model = VGG16(include_top=False, 
            pooling="max", 
            input_shape= (224, 224, 3))


# mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False #resetting 
# we only wanna update the classification layer in the end,
# so now we "freeze" all weigths in the feature extraction part and make them "untrainable"



########## adding classification layers #########

# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
bn = BatchNormalization()(flat1) #normalize the feature weights 
# 1st layer
class1 = Dense(256, 
            activation="relu")(bn)
# 2nd layer               
class2 = Dense(128, 
            activation="relu")(class1)

# 3rd layer               
#class3 = Dense(90, 
#            activation="relu")(class2)
# 4th layer               
#class4 = Dense(30, 
#            activation="relu")(class3)
# output layer    
output = Dense(7, #  7 lables
            activation="softmax")(class2) 

# define new model
model = Model(inputs=model.inputs, 
            outputs=output)

# compile
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate= 0.01,
    decay_steps=10000,
    decay_rate=0.9)
sgd = SGD(learning_rate=lr_schedule)

model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'])




############## FIT & TRAIN #################

#model
skin_cancer_classifier = model.fit(train_ds,
                    steps_per_epoch= train_ds.samples // batch_size,
                    epochs = n_epochs,
                    validation_data=train_ds,
                    validation_steps= val_ds.samples // batch_size,
                    batch_size = batch_size,
                    verbose = 1)#,
                    #callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights = True)]
                    #)



############ EVALUATION #####################
hf.plot_history(skin_cancer_classifier, n_epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

################### MODEL PREDICT ########################
predictions = model.predict(test_ds, # X_test
                            batch_size=batch_size)


# Make classification report
report=(classification_report(test_ds.classes, # y_test 
                                            predictions.argmax(axis=1),
                                            target_names=test_ds.class_indices.keys())) #labels

print(report)
# Define outpath for classification report
outpath_report = os.path.join(os.getcwd(), "out", "VGG16_skin_cancer_report.txt")

# Save the  classification report
file = open(outpath_report, "w")
file.write(report)
file.close()

print( "Saving the skin cancer classification report in the folder ´out´")


