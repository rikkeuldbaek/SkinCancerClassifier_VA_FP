### Final Project - Skin Cancer Classifier
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 8th of May 2023

#--------------------------------------------------------#
################ SKIN CANCER CLASSIFIER ##################
#--------------------------------------------------------#

# (please note that some of this code has been adapted from https://www.kaggle.com/code/gpiosenka/efficientnetb5-f1-score-83)

#loading packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import argparse

###################### PARSER ############################
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()

    #add arguments for data.py
    parser.add_argument("--train_split_bal", type=float, default= .70, help= "Specify train split for balanced data.") 
    parser.add_argument("--val_split_bal", type=float, default= .15, help= "Specify validation split for balanced data.") 
    parser.add_argument("--train_split_ubal", type=float, default= .70, help= "Specify train split for unbalanced data.") 
    parser.add_argument("--val_split_ubal", type=float, default= .15, help= "Specify validation split for unbalanced data.") 

    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments


################# DF PREPROCESSING THE GROUNDTRUTH.CSV #################

def data_preprocessing():

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
    
    return df


#################### CREATE BALANCED DATA #####################
# sampling to make balanced data
def balance_data(df):
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

    return lil_df



################# TEST AND TRAIN SPLIT - BALANCED DATA ##################

def split_balanced_data(lil_df, train_split_bal, val_split_bal):
    # print balanced data and label distribution
    print( "***"*15)
    print('Number of data points in balanced data: ', len(lil_df))
    print('Balanced label distribution: ')  
    print (lil_df['label'].value_counts())  
    print( "***"*15)
    print( "  "*15)

    train_split= train_split_bal # percentage of data used for training, default .70
    valid_split= val_split_bal # percentage of data used for validation, default .15

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
    print( "  "*15)

    return(train_df_bal, test_df_bal, val_df_bal)



################## TEST AND TRAIN SPLIT - UNBALANCED DATA #################

def split_unbalanced_data(df, train_split_ubal, val_split_ubal):
    # Current label distribution (unbalanced)
    print( "***"*15)
    print('Number of data points in unbalanced data: ', len(df))
    print('Unbalanced label distribution: ')
    print(df['label'].value_counts()) 
    print( "***"*15)
    print( "  "*15)

    train_split= train_split_ubal # percentage of data used for training, default .70
    valid_split= val_split_ubal # percentage of data used for validation, default .15

    # percentage of data used for test is 1-train_split-valid_split 
    test_val_split = valid_split/(1-train_split) # split of 0.5 

    # Splitting data into train df and remaining data
    train_df_ubal, remaining_data_ubal = train_test_split(df, train_size=train_split, shuffle=True, random_state=188)

    # Splitting remaining data into validation and test df
    val_df_ubal, test_df_ubal=train_test_split(remaining_data_ubal, train_size= test_val_split, shuffle=True, random_state=188)

    # Printing splits
    print( "***"*15)
    print('Current split of unbalanced data: ')  
    print( "---"*15)
    print('train_df length: ', len(train_df_ubal))  
    print( "---"*15)
    print('test_df length: ', len(test_df_ubal))
    print( "---"*15)
    print( 'valid_df length: ', len(val_df_ubal))
    print( "***"*15)

    return train_df_ubal, test_df_ubal, val_df_ubal


#################### MAIN FUNCTION #######################
def main():
    args = input_parse()

    # functions for data preprocessing
    df = data_preprocessing()
    lil_df = balance_data(df)

    # balanced data
    train_df_bal, test_df_bal, val_df_bal = split_balanced_data(lil_df, args.train_split_bal, args.val_split_bal)

    # unbalanced data
    train_df_ubal, test_df_ubal, val_df_ubal = split_unbalanced_data(df, args.train_split_ubal, args.val_split_ubal)

if __name__ == '__main__':
    main()
