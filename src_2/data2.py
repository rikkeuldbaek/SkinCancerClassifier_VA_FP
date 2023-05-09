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

# Data visualization libraries
import matplotlib.pyplot as plt
import cv2

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


#################### FIX UNBALANCED TRAIN DATA #####################
# Current label distribution
print( "***"*15)
print('Number of data points: ', len(df))
print('Unbalanced label distribution: ')
print(df['label'].value_counts()) 
print( "***"*15)
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
df=pd.concat(samples, axis=0).reset_index(drop=True)

print( "***"*15)
print('Number of data points: ', len(df))
print('Balanced label distribution: ')  
print (df['label'].value_counts())  
print( "***"*15)
print( "  "*15)



##################### TEST AND TRAIN SPLIT #####################

train_split=.80 # percentage of data used for training
valid_split=.10 # percentage of data used for validation

# percentage of data used for test is 1-train_split-valid_split 
test_val_split = valid_split/(1-train_split) # split of 0.5 

# Splitting data into train df and remaining data
train_df, remaining_data = train_test_split(df, train_size=train_split, shuffle=True, random_state=1)

# Splitting remaining data into validation and test df
val_df, test_df=train_test_split(remaining_data, train_size= test_val_split, shuffle=True, random_state=1)

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


 
########## PLOTTING THE SAMPLE ########## 
# Isolate a sample of each diagnostic category (label)
group=df.groupby('label')
samples = group.sample(1, seed)

# Convert image column to be a full path
def convert_image_path(image_path):
    base_dir = os.path.join(os.getcwd(),"data","archive","images")
    return os.path.join(base_dir, image_path)

samples['image'] = samples['image'].apply(convert_image_path)


# Display 16 picture of the dataset with their labels
#random_index = np.random.randint(10000, 90000, 7)
#fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10),
                        #subplot_kw={'xticks': [], 'yticks': []})


#for i, ax in enumerate(axes.flat):
#    ax.imshow(plt.imread(samples.image[random_index[i]]))
#    ax.set_title(samples.label[random_index[i]])
#plt.tight_layout()
#plt.show()
#plt.savefig('sample_pngs/diagnostic_categories.png') #save figures in folder "sample_pngs"

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.subplot(3, 3, 1)
plt.imshow(plt.imread(samples['image'].iloc[0]))
plt.subplot(3, 3, 2)
plt.imshow(plt.imread(samples['image'].iloc[1]))
plt.subplot(3, 3, 3)
plt.imshow(plt.imread(samples['image'].iloc[2]))
plt.subplot(3, 3, 4)
plt.imshow(plt.imread(samples['image'].iloc[3]))
plt.show()
#######nxt row
plt.subplot(3, 3, 5)
plt.imshow(plt.imread(samples['image'].iloc[4]))
plt.subplot(3, 3, 6)
plt.imshow(plt.imread(samples['image'].iloc[5]))
plt.subplot(3, 3, 7)
plt.imshow(plt.imread(samples['image'].iloc[6]))

plt.show()


plt.savefig('sample_pngs/diagnostic_categories.png') #save figures in folder "sample_pngs"

