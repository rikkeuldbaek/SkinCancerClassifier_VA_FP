
# import packages
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


################## TEST DATA SET #####################

# Generate data paths with labels
data_directory = os.path.join(os.getcwd(),  "skin_data", "Testing")
test_file_paths = []
test_labels = []


diagnosis_folders = os.listdir(data_directory)

for diagnosis in diagnosis_folders: 
    diagnosis_folder_path = os.path.join(data_directory, diagnosis) #folderpath to cancer /non_cancer folders
    images = os.listdir(diagnosis_folder_path)

    #make filepath for every image
    for image in images:
        image_path = os.path.join(diagnosis_folder_path, image)
        test_file_paths.append(image_path)
        test_labels.append(diagnosis) 


# Merge data paths with labels into a test dataframe
Fseries = pd.Series(test_file_paths, name= 'filepaths')
Lseries = pd.Series(test_labels, name='labels')
test_df = pd.concat([Fseries, Lseries], axis= 1)




################## TRAINING DATA SET #####################
# Generate data paths with labels
data_directory = os.path.join(os.getcwd(), "skin_data", "Training")
train_file_paths = []
train_labels = []


diagnosis_folders = os.listdir(data_directory)

for diagnosis in diagnosis_folders: 
    diagnosis_folder_path = os.path.join(data_directory, diagnosis) #folderpath to cancer /non_cancer folders
    images = os.listdir(diagnosis_folder_path)

    #make filepath for every image
    for image in images:
        image_path = os.path.join(diagnosis_folder_path, image)
        train_file_paths.append(image_path)
        train_labels.append(diagnosis)


# Merge data paths with labels into a training dataframe
Fseries = pd.Series(train_file_paths, name= 'filepaths')
Lseries = pd.Series(train_labels, name='labels')
train_df = pd.concat([Fseries, Lseries], axis= 1)


#print(train_df)
#print(test_df)

