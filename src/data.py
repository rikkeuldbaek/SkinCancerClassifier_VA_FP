
# import packages
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")




# Generate data paths with labels
data_directory = os.path.join(os.getcwd(), "skin_data")
filepaths = []
labels = []

diagnosis_folders = os.listdir(data_directory)

for diagnosis in diagnosis_folders: 
    foldpath = os.path.join(data_directory, diagnosis) #folderpath to cancer /non_cancer folders
    sub_folders = os.listdir(foldpath) #testing / training folders within above two folders
    
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(foldpath, sub_folder)
        image_list = os.listdir(sub_folder_path)
    
        for image in image_list:
            image_path = os.path.join(sub_folder_path, image)
            filepaths.append(image_path)
            labels.append(diagnosis) #cancer / non_cancer

#Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis= 1)
print(df["filepaths"])



