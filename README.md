# **Final Project - Skin Cancer Classifier**
## **Cultural Data Science - Visual Analytics** 
#### Author: Rikke Uldbæk (202007501)
#### Date: 8th of May 2023
<br>

## **1.1 Contributors**
The only contributor of this assignment is the author of this project (Rikke Uldbæk, 202007501). The following link is a link to the GitHub repository of this project: 

https://github.com/rikkeuldbaek/SkinCancerClassifier_VA_FP.git

<br>

# **1.2 Description**

For this final project I have worked with a collection of dermatoscopic images with different categories of pigmented skin lesions, i.e., skin cancer, in order to build a classifier that is able to segment and classify different diagnostic categories of skin cancer. This repository contains source code which trains a *pretrained CNN* on the skin cancer dataset (*the HAM10000 dataset*), classifies the diagnostic categories of skin cancer and produces a classification report and a training/validation history plot, in order to evaluate the performance of the classifier. 



<br>

# **1.3 Methods**
**VGG16**

<br>

**Data Augmentation**

<br>

# **1.4 Data**
For this final project I have used the *HAM10000 dataset* ("Human Against Machine with 10000 training images"). This dataset contains 10015 multi-source dermatoscopic images of common pigmented skin lesions, also known as skin cancer. This collection of images of skin cancer includes seven representative diagnostic categories within the domain of pigmented lessions. These seven categories are: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (AKIEC), basal cell carcinoma (BCC), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, BKL), dermatofibroma (df), melanoma (MEL), melanocytic nevi (NV) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, VASC) (Tschandl, 2018). Furthermore, the *HAM10000 dataset* consists of a *Ground Truth* .csv file matching each image filename to its diagnostic category. The plot below illustrates each of the seven diagnostic categories of skin cancer.

#### **Diagnostic Categories of Skin Cancer** 
![Diagnostic Categories of Skin Cancer](sample_pngs/diagnostic_categories.png)


The distribution of the diagnostic categories of skin cancer was intitially rather unbalanced, as seen from plot 1. For instance the data contained 6705 images of the diagnostic category *NV*, while containing 115 images of the diagnostic category *DF*. Thus, a maximum limit of 500 samples per diagnostic category was established, although some categories had less than 500 data points. This resulted in a slightly more balanced dataset, however it is not completely balanced out as visually evident from plot 2. Furthermore, this maximum limit of 500 samples per diagnostic category decreased the sample size substantially from 10015 images to 2584 images, which may have a great impact on the modelling. 

![Unbalanced data](sample_pngs/unbalanced_distribution.png)

![Balanced Data](sample_pngs/balanced_distribution.png)



<br>


# **1.5 Repository Structure**
The scripts of this project require a certain folder structure, thus the table below presents the required folders and their description and content.

|Folder name|Description|Content|
|---|---|---|
|```data```|images of skin cancer and .csv file with labels|```archive/images/all_images```,```archive/GroundTruth.csv```|
|```src```|model, data, and plot scripts|```data.py```, ```classifier.py```, ```plot.py```, ```unzip_data.py```|
|```out```|classification reports and training/validation history plot|```classification_report.txt```, ```train_val_history_plot.png```|
|```readme_pngs```|plots for the readme file|```balanced_distribution.png```, ```unbalanced_distribution.png```, ```diagnostic_categories.png```|
|```utils```|helper functions|```helper_func.py```|

The ```unzip_data.py``` script located in ```src``` unzips the data into the ```data```folder. The ```data.py``` script located in ```src``` preprocesses the data and produces training, test, and validation data. The ```classifier.py``` script located in ```src``` produce a skin cancer classifier model, a classification report and a training/validation history plot which are saved in the folder ```out```. Helper functions for plotting are found in the folder ```utils``` and relevant plots for illustrations are found in the folder ```readme_pngs```. 


<br>

# **1.6 Usage and Reproducibility**
## **1.6.1 Prerequsities** 
In order for the user to be able to run the code, please make sure to have bash and python 3 installed on the used device. The code has been written and run with Python 3.9.2 on a Mac computer. In order to run the provided code for this assignment, please follow the instructions below.

<br>

## **1.6.2 Setup Instructions** 
**1) Clone the repository**
Please execute the following command in the terminal to clone this repository. 
```python
git clone https://github.com/rikkeuldbaek/SkinCancerClassifier_VA_FP.git
 ```

 **2) Setup** <br>
Please execute the following command in the terminal to setup a virtual environment (```VA_fp_env```) and install packages.
```python
bash setup.sh
```
<br>

**3) Download the data and unzip the file** <br>
Please download the *HAM10000 dataset* from Kaggle ([HAM10000 dataset](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)
), and store the ```archive.zip``` zipfile in the ```data``` folder in this repository. Please run the following script in order to unzip file into the ```data``` folder. The command must be executed in the terminal. This may take a few minutes since the files in the zipfile are around 3GB in total. 

```python
python src/unzip_data.py 
```

<br>

## **1.6.3 Running the scripts** 
Please execute the following command in the terminal to automatically run the ```data.py``` and ```classifier.py``` scripts.
```python
bash run.sh
```


## **1.6.4) Script arguments**
The skin cancer classifier have the following default arguments stated in the table below. These arguments can be modified and adjusted in the ```run.sh``` script. If no modifications is added, default parameters are run. In case help is needed, please use ```--help``` after an argument. 
```python
# run the code
python src/classifier.py #add arguments here
```

<br>

The ```classifier.py``` takes the following arguments:
|Argument|Type|Default|
|---|---|---|
|--|| |
|--|| |
|--|| |
|--|| |
|--|| |
|--|| |

<br>

The ```data.py``` takes the following arguments:
|Argument|Type|Default|
|---|---|---|
|--|| |


<br>

### **Important to note** <br>
The target_size and the input_shape argument must be specified _without_ commas in the ```run.sh``` script, please see following command for an example of such:

```python
python src/classifier.py --target_size  224 224 --input_shape 224 224 3
 ```

Similarly it is very important to note that the ```data.py``` is automatically called upon when running the ```classifier.py``` script, thus the arguments for ```data.py``` must be parsed to the ```lassifier.py``` script in ```run.sh```:

````python 
python src/classifier.py --classifier_arguments --data_arguments
````


# **1.7 Results**

**[Classification report](out/classification_report.txt)** (open link to see) <br>
The model shows an accuracy of 76%. Furthermore, the model seems to be best at classifying *blouses*, *nehru jackets*, *lehenga*, and *mojaris for men* with F1 scores of 90%, 85%, 84%, 84% respectively. 


**Training and validation history plot**

![Training and validation history plot](out/train_val_history_plot.png)


<br>


<br>

## **Resources**
[HAM10000 dataset](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)


Tschandl, P. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (ViDIR Group, Ed.; V4 ed.). Harvard Dataverse. https://doi.org/10.7910/DVN/DBW86T