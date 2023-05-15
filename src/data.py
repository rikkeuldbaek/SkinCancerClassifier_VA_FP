### Skin Cancer Classifier
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 8th of May 2023

#--------------------------------------------------------#
################ SKIN CANCER CLASSIFIER ##################
#--------------------------------------------------------#

#loading packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

################# DF PREPROCESSING THE GROUNDTRUTH.CSV #################

#loading in Ground Truth .csv
df=pd.read_csv(os.path.join(os.getcwd(),"data","archive","GroundTruth.csv"))

# Labels of diagnostic skin cancer types
labels=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'] 

# Insert missing .jpg after image filename
df['image']=df['image'].apply(lambda x: x+ '.jpg')

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


#################### CREATE BALANCED DATA #####################
# sampling to make balanced data
size=500 #sample if a class has more than 500 data points 
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
lil_df=pd.concat(samples, axis=0).reset_index(drop=True)



################# TEST AND TRAIN SPLIT - BALANCED DATA ##################
# print balanced data and label distribution
print( "***"*15)
print('Number of data points in balanced data: ', len(lil_df))
print('Balanced label distribution: ')  
print (lil_df['label'].value_counts())  
print( "***"*15)
print( "  "*15)
train_split=.70 # percentage of data used for training
valid_split=.15 # percentage of data used for validation

# percentage of data used for test is 1-train_split-valid_split 
test_val_split = valid_split/(1-train_split) # split of 0.5 

# Splitting data into train df and remaining data
train_df_bal, remaining_data_bal = train_test_split(lil_df, train_size=train_split, shuffle=True, random_state=188)

# Splitting remaining data into validation and test df
val_df_bal, test_df_bal =train_test_split(remaining_data_bal, train_size= test_val_split, shuffle=True, random_state=188)

# Printing splits
print( "***"*15)
print('Current split of balanced data: ')  
print( "---"*15)
print('train_df length: ', len(train_df_bal))  
print( "---"*15)
print('test_df length: ', len(test_df_bal))
print( "---"*15)
print( 'valid_df length: ', len(val_df_bal))
print( "***"*15)


################## TEST AND TRAIN SPLIT - UNBALANCED DATA #################
# Current label distribution (unbalanced)
print( "***"*15)
print('Number of data points in ubalanced data: ', len(df))
print('Unbalanced label distribution: ')
print(df['label'].value_counts()) 
print( "***"*15)

train_split=.70 # percentage of data used for training
valid_split=.15 # percentage of data used for validation

# percentage of data used for test is 1-train_split-valid_split 
test_val_split = valid_split/(1-train_split) # split of 0.5 

# Splitting data into train df and remaining data
train_df_ubal, remaining_data_ubal = train_test_split(df, train_size=train_split, shuffle=True, random_state=188)

# Splitting remaining data into validation and test df
val_df_ubal, test_df_ubal=train_test_split(remaining_data_ubal, train_size= test_val_split, shuffle=True, random_state=188)

# Printing splits
print( "***"*15)
print('Current split of ubalanced data: ')  
print( "---"*15)
print('train_df length: ', len(train_df))  
print( "---"*15)
print('test_df length: ', len(test_df))
print( "---"*15)
print( 'valid_df length: ', len(val_df))
print( "***"*15)