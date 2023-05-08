
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
#import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report



################# DF PREPROCESSING THE GROUNDTRUTH.CSV #################

#loading in Ground Truth .csv
df=pd.read_csv(os.path.join(os.getcwd(),"data","archive","GroundTruth.csv"))

# Labels of diagnostic skin cancer types
labels=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'] 

# Insert missing .jpg after image filename
df['image']=df['image'].apply(lambda x: x+ '.jpg')
print (df.head())

# Rearrange dataframe and make a label column with type of skin cancer
label_list=[]
for i in range (len(df)):
    row= list(df.iloc[i])
    del row[0]
    index=np.argmax(row)
    label=labels[index]
    label_list.append(label)
df['label']= label_list
df=df.drop(labels, axis=1)
print (df.head())

##################### TEST AND TRAIN SPLIT #####################

train_split=.80 # percentage of data used for training
valid_split=.10 # percentage of data used for validation

# percentage of data used for test is 1-train_split-valid_split 
test_val_split = valid_split/(1-train_split) # split of 0.5 

# Splitting data into train df and remaining data
train_df, remaining_data = train_test_split(df, train_size=train_split, shuffle=True, random_state=666)

# Splitting remaining data into validation and test df
val_df, test_df=train_test_split(remaining_data, train_size= test_val_split, shuffle=True, random_state=666)

# Printing splits
print( "***"*15)
print('Current data split: ')  
print( "---"*15)
print('train_df length: ', len(train_df))  
print( "---"*15)
print('test_df length: ', len(test_df))
print( "---"*15)
print( 'valid_df length: ', len(val_df))
print( "***"*15)


# The data is unbalanced with regards to labels 
print(train_df['label'].value_counts())
print(test_df['label'].value_counts())
print(val_df['label'].value_counts())


##################### FIX UNBALANCED DATA #####################
print ('original number of classes: ', len(df['label'].unique()))
size=300 # set number of samples for each class
samples=[]
group=df.groupby('label')
for label in df['label'].unique():
    Lgroup=group.get_group(label)
    count=int(Lgroup['label'].value_counts())    
    if count>=size:
        sample=Lgroup.sample(size, axis=0)        
    else:        
        sample=Lgroup.sample(frac=1, axis=0)
    samples.append(sample) 
train_df=pd.concat(samples, axis=0).reset_index(drop=True)
print (len(train_df))
print ('final number of classes: ', len(train_df['label'].unique()))       
print (train_df['label'].value_counts())  
