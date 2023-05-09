### Skin Cancer Classifier
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 8th of May 2023

#--------------------------------------------------------#
################ SKIN CANCER CLASSIFIER ##################
#--------------------------------------------------------#

#loading packages
import os

# Data visualization libraries
import matplotlib.pyplot as plt
import cv2

# import data
import data as dt

# load in data
df = dt.df
lil_df = dt.lil_df


################## PLOTTING THE SAMPLE ##################
# Isolate a sample of each diagnostic category (label)
group=df.groupby('label')
samples = group.sample(1, random_state =6)

# Convert image column to be a full path
def convert_image_path(image_path):
    base_dir = os.path.join(os.getcwd(),"data","archive","images")
    return os.path.join(base_dir, image_path)

samples['image'] = samples['image'].apply(convert_image_path)


# Display the 7 diagnostic categories of the dataset with their labels
plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams["figure.autolayout"] = False
#### PLOTTING EACH IMAGE ####
plt.subplot(3, 3, 1)
plt.title("Diagnostic Categories of Skin Cancer")
plt.imshow(plt.imread(samples['image'].iloc[0]))
plt.title(samples['label'].iloc[0])
plt.subplot(3, 3, 2)
plt.imshow(plt.imread(samples['image'].iloc[1]))
plt.title(samples['label'].iloc[1])
plt.subplot(3, 3, 3)
plt.imshow(plt.imread(samples['image'].iloc[2]))
plt.title(samples['label'].iloc[2])
plt.subplot(3, 3, 4)
plt.imshow(plt.imread(samples['image'].iloc[3]))
plt.title(samples['label'].iloc[3])
plt.show()
plt.subplot(3, 3, 5)
plt.imshow(plt.imread(samples['image'].iloc[4]))
plt.title(samples['label'].iloc[4])
plt.subplot(3, 3, 6)
plt.imshow(plt.imread(samples['image'].iloc[5]))
plt.title(samples['label'].iloc[5])
plt.subplot(3, 3, 7)
plt.imshow(plt.imread(samples['image'].iloc[6]))
plt.title(samples['label'].iloc[6])
plt.grid(False)
plt.show()
plt.savefig('sample_pngs/diagnostic_categories.png') #save figures in folder "sample_pngs"
plt.close()



################## PLOTTING THE DISTRIBUTION OF LABELS THE DATA #################

# Unbalanced data
ax = df['label'].value_counts().plot(kind='bar',
                                    figsize=(10,8),
                                    color=['#9b5fe0','#f9a52c', '#16a4d8', '#60dbe8', '#8bd346', '#efdf48', '#d64e12' ])
ax.set_title("Unbalanced Distribution of Diagnostic Categories", fontsize=20)
ax.set_xlabel("Diagnostic Categories", fontsize=14)
ax.set_ylabel("Frequency",fontsize=14)
plt.show()
plt.savefig('sample_pngs/unbalanced_distribution.png') #save figures in folder "sample_pngs"
plt.close()


# Balanced data
ax = lil_df['label'].value_counts().plot(kind='bar',
                                    figsize=(10,8),
                                    color=['#9b5fe0','#f9a52c', '#16a4d8', '#60dbe8', '#8bd346', '#efdf48', '#d64e12' ])
ax.set_title("Balanced Distribution of Diagnostic Categories", fontsize=20)
ax.set_xlabel("Diagnostic Categories", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
plt.show()
plt.savefig('sample_pngs/balanced_distribution.png') #save figures in folder "sample_pngs"
plt.close()