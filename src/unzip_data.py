### Final Project - Skin Cancer Classifier
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 8th of May 2023

#--------------------------------------------------------#
###################### UNZIP DATA ########################
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

#import zipfile module and pandas
from zipfile import ZipFile
import os

##################### UNZIPPING #######################

# define path for zipfile of images
skin_cancer_path = os.path.join(os.getcwd(), "data", "archive.zip")
     

# reading in files
with ZipFile(skin_cancer_path, mode="r") as archive:
    badfiles=archive.testzip()
    print(badfiles) #print if any files are corrupted
    archive.extractall("data/")


##################### COUNTING DATA #######################
# define path of images
path = os.path.join(os.getcwd(),"data","archive","images") 

# check all data is loaded in
count = 0

# Iterate directory
for i in os.listdir(path):
    # check if current path is a file
    if os.path.isfile(os.path.join(path, i)):
        count += 1
print('File count:', count) # 10017 images

    