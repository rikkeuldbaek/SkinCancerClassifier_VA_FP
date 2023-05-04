### Skin Cancer Model
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 27th of April 2023

#--------------------------------------------------------#
###################### UNZIP DATA ########################
#--------------------------------------------------------#

#import zipfile module and pandas
from zipfile import ZipFile
import os
#import pandas as pd

#define path for zipfile of images
skin_cancer_path="/work/VA_final_project/skin_cancer/archive.zip"
skin_cancer_path2="/work/VA_final_project/skin_cancer/archive_new.zip"

#unzip  and extract in current directory
#with ZipFile(skin_cancer_path, 'r') as f:
#    f.extractall()



filename = skin_cancer_path

import zipfile

#try:
#    z = zipfile.ZipFile(filename)
#except zipfile.BadZipfile:
#    import commands
#    commands.getoutput('zip -FF '+filename)
#    z = zipfile.ZipFile(filename)

#for i in z.infolist():
#    print(i.filename, i.file_size)


#try: 
#    z.read('images/ISIC_0026299.jpg')
#except zipfile.BadZipfile:
#    print('Bad CRC-32')


#with zipfile.ZipFile(skin_cancer_path, mode="r") as archive:
#    archive.printdir()


#see if there are any corrupted files
try:
    with zipfile.ZipFile(skin_cancer_path2) as archive:
        archive.printdir()
except zipfile.BadZipFile as error:
    print(error) #seems like there are not 

# print info of a file
#with zipfile.ZipFile(skin_cancer_path, mode="r") as archive:
#     info = archive.getinfo("images/ISIC_0026299.jpg")
#     print(info)
     

####### reading in files
with zipfile.ZipFile(skin_cancer_path, mode="r") as archive:
    badfiles=archive.testzip()
    print(badfiles)
    #archive.extractall("data/")


import subprocess
import zipfile

zin = zipfile.ZipFile (skin_cancer_path, 'r')
zout = zipfile.ZipFile (skin_cancer_path2, 'w')
for item in zin.infolist():
    buffer = zin.read(item.filename)
    print(buffer)
    if (item.filename[-4:] != '0026299.jpg'):
        zout.writestr(item, buffer)
zout.close()
zin.close()

#trying to extract new files minus deleted one
with zipfile.ZipFile(skin_cancer_path, mode="r") as archive:
    archive.extractall("data/") #save in data folder



#filename = skin_cancer_path

#import zipfile

#z = zipfile.ZipFile(filename)

#for i in z.infolist():
#    print i.filename, i.file_size

#z.read('somefile')







#images/ISIC_0026299.jpg' 

