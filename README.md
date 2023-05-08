# **Final Project - Skin Cancer Classifier**
## **Cultural Data Science - Visual Analytics** 
#### Author: Rikke Uldbæk (202007501)
#### Date: 8th of May 2023
<br>

## **1.1 Contributors**
The only contributor of this assignment is the author of this project (Rikke Uldbæk, 202007501). The following link is a link to the GitHub repository of this project: 

https://github.com/rikkeuldbaek/SkinCancerClassifier_VA_FP 

<br>

# **1.2 Description**

For this final project I have worked with a collection of dermatoscopic images with different types of pigmented skin lesions, i.e., skin cancer, in order to build a classifier that is able to segment and classify different types of skin cancer. This repository contains source code which trains a *pretrained CNN* on the skin cancer dataset (*the HAM10000 dataset*), classifies the skin cancer types and produces a classification report and a training/validation history plot, in order to evaluate the performance of the classifier. 



<br>

# **1.3 Methods**
**VGG16**

<br>

**Data Augmentation**

<br>

# **1.4 Data**
For this final project I have used the *HAM10000 dataset* ("Human Against Machine with 10000 training images"). This dataset contains 10015 multi-source dermatoscopic images of common pigmented skin lesions, also known as skin cancer. This collection of images of skin cancer includes seven representative diagnostic categories within the domain of pigmented lessions. These seven categories are: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (AKIEC), basal cell carcinoma (BCC), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, BKL), dermatofibroma (df), melanoma (MEL), melanocytic nevi (NV) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, VASC) **KILDE**. Furthermore, the *HAM10000 dataset* consists of a *Ground Truth* .csv file matching each image filename to its diagnostic category.

The distribution of data within 


<br>

# **1.5 Repository Structure**

<br>

# **1.6 Usage and Reproducibility**
## **1.6.1 Prerequsities** 
In order for the user to be able to run the code, please make sure to have bash and python 3 installed on the used device. The code has been written and run with Python 3.9.2 on a Mac computer. In order to run the provided code for this assignment, please follow these instructions:

<br>

## **1.6.2 Setup Instructions** 
**1) Clone the repository**
```python
git clone https://github.com/AU-CDS/assignment1-simple-image-search-rikkeuldbaek
 ```

 **2) Setup** <br>
Setup virtual environment (```VA1_env```) and install packages.
```python
bash setup.sh
```
<br>

## **1.6.3 Running the scripts** 
### **1.6.3.1) Run the Simple Image Search Algorithm**
Please open the folder ```notebook``` and open the ```simple_image_search.ipynb``` script and press "Run All". The *target image* can be modified in chunk 3:
````ipynb
compare_top5_hist("target_image")
````

<br>

### **1.6.3.2) Run the Complex Image Search Algorithm** 
The command below will automatically run the ```image_search_VGG16_KNearestNeigh.py```script and produce the described results.
```python
bash run.sh
```

However, for the user to change the filename of the *target image* please specify this in the ```run.sh``` script:
```python
python3 src/image_search_VGG16_KNearestNeigh.py --target_flower_image user_specific_target_image
```

### **1.6.3) Script arguments**

The ```image_search_VGG16_KNearestNeigh.py``` takes the following arguments:
|Argument|Type|Default|
|---|---|---|
|--target_flower_image|str|image_0021.jpg|


<br>


# **1.7 Results**


<br>

## **1.7.1 Results of the Simple Image Search Algorithm**

<br>

## **1.7.2 Results of the Complex Image Search Algorithm**
From the results of the complex image search algorithm, three out of five 


<br>

## **Resources**
[HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
INSERT ZOTERO LINK FOR DATA



# DATA
The labels were highly unbalanced eg. data contained 6705 images of cancer type  NV, while containing 115 images of cancer type DF. Thus the data was balanced out, however this decreased the sample size drastically.


