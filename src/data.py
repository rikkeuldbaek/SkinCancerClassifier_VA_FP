
# import packages
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")



################## TEST DATA SET #####################

# Generate data paths with labels
data_directory = os.path.join(os.getcwd(), "skin_data", "Testing")
file_paths = []
labels = []


diagnosis_folders = os.listdir(data_directory)

for diagnosis in diagnosis_folders: 
    diagnosis_folder_path = os.path.join(data_directory, diagnosis) #folderpath to cancer /non_cancer folders
    images = os.listdir(diagnosis_folder_path)

    #make filepath for every image
    for image in images:
        image_path = os.path.join(diagnosis_folder_path, image)
        file_paths.append(image_path)
        labels.append(diagnosis)


#Concatenate data paths with labels into one dataframe
Fseries = pd.Series(file_paths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
test_df = pd.concat([Fseries, Lseries], axis= 1)




################## TRAINING DATA SET #####################
# Generate data paths with labels
data_directory = os.path.join(os.getcwd(), "skin_data", "Training")
file_paths = []
labels = []


diagnosis_folders = os.listdir(data_directory)

for diagnosis in diagnosis_folders: 
    diagnosis_folder_path = os.path.join(data_directory, diagnosis) #folderpath to cancer /non_cancer folders
    images = os.listdir(diagnosis_folder_path)

    #make filepath for every image
    for image in images:
        image_path = os.path.join(diagnosis_folder_path, image)
        file_paths.append(image_path)
        labels.append(diagnosis)


#Concatenate data paths with labels into one dataframe
Fseries = pd.Series(file_paths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
train_df = pd.concat([Fseries, Lseries], axis= 1)
print(test_df)
print(train_df)
