### Final Project - Skin Cancer Classifier
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 27th of April 2023

#--------------------------------------------------------#
################ SKIN CANCER CLASSIFIER ##################
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

# Install packages
# path tools
import os, sys
 
# data 
import data as dt

# tf tools 
import tensorflow as tf
 
# image processsing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# call backs
from tensorflow.keras.callbacks import EarlyStopping

#scikit-learn
from sklearn.metrics import classification_report


# Import predefined helper functions (for plotting)
sys.path.append(os.path.join("utils"))
import helper_func as hf
 
# Scripting
import argparse

###################### PARSER ############################
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()

    #add arguments for data.py
    parser.add_argument("--train_split_ubal", type=float, default= .70, help= "Specify train split for unbalanced data.") 
    parser.add_argument("--val_split_ubal", type=float, default= .15, help= "Specify validation split for unbalanced data.")
    parser.add_argument("--target_size",nargs='+', type=int, default= (224, 224), help= "Specify target size for image preprocessing.") 
    parser.add_argument("--horizontal_flip", type=bool, default= True, help= "Specify wether the image should be flipped horizontally when agumented.") 
    parser.add_argument("--vertical_flip", type=bool, default= True, help= "Specify wether the image should be flipped vertically when agumented.") 
    parser.add_argument("--zca_whitening", type=bool, default= True, help= "Specify whether to reduce the redundancy in the matrix of pixel images when augmented.") 
    parser.add_argument("--shear_range", type=float, default= 0.2, help= "Specify the shear angle in counter-clockwise direction in degrees when augmented.") 
    parser.add_argument("--zoom_range", type=float, default= 0.2, help= "Specify range for random zoom when augmented.") 
    parser.add_argument("--rotation_range", type=int, default= 20, help= "Specify degree range for random rotations when augmented.") 
    parser.add_argument("--rescale_1", type=float, default= 1. , help= "Specify ( first digit ) rescaling factor when augmented.") 
    parser.add_argument("--rescale_2", type=float, default= 255. , help= "Specify ( second digit ) rescaling factor when augmented.") 
    parser.add_argument("--batch_size", type=int, default= 30 , help= "Specify size of batch.") 
    parser.add_argument("--n_epochs", type=int, default= 30, help= "Specify number of epochs for model training.") 
    parser.add_argument("--class_mode", type=str, default= "categorical" , help= "Specify class type of target values.") 
    parser.add_argument("--pooling", type=str, default= "avg" , help= "Specify pooling mode for feature extraction.") 
    parser.add_argument("--input_shape",nargs='+', type=int, default= (224, 224, 3) , help= "Specify shape of tuple for feature extraction.") 
    parser.add_argument("--monitor", type=str, default= 'val_loss' , help= "Specify quantity to be monitored.") 
    parser.add_argument("--patience", type=int, default= 5, help= "Specify number of epochs with no improvement after which training will be stopped.") 
    parser.add_argument("--restore_best_weights", type=bool, default= True, help= "Specify whether to restore model weights from the epoch with the best value of the monitored quantity.") 
    parser.add_argument("--nodes_layer_1", type=int, default= 300, help= "Specify number of nodes in first hidden layer.") 
    parser.add_argument("--nodes_layer_2", type=int, default= 150, help= "Specify number of nodes in second hidden layer.") 
    parser.add_argument("--activation_hidden_layer", type=str, default= "relu", help= "Specify activation function to use in hidden layers.") 
    parser.add_argument("--activation_output_layer", type=str, default= "softmax", help= "Specify activation function to use in output layer.") 
    parser.add_argument("--initial_learning_rate", type=float, default= 0.01, help= "Specify the initial learning rate.") 
    parser.add_argument("--decay_steps", type=int, default= 10000, help= "Specify number of decay steps.") 
    parser.add_argument("--decay_rate", type=float, default= 0.9, help= "Specify the decay rate.") 
    parser.add_argument("--loss", type=str, default= "categorical_crossentropy", help= "Specify loss function.") 


    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments

#################### LOADING DATA ####################

#load in data
args = dt.input_parse()
df = dt.data_preprocessing()
train_df_ubal, test_df_ubal, val_df_ubal = dt.split_unbalanced_data(df, args.train_split_ubal, args.val_split_ubal)
train_df, test_df, val_df = train_df_ubal, test_df_ubal, val_df_ubal


#################### IMAGE DATA GENERATOR ####################

def data_augmentation(train_df, test_df, val_df, horizontal_flip, vertical_flip, zca_whitening, shear_range,zoom_range, rotation_range, rescale_1, rescale_2, batch_size, class_mode, target_size ):

    # Specify Image Data Generator 
    datagen=ImageDataGenerator(horizontal_flip= horizontal_flip,
                                vertical_flip=vertical_flip,
                                zca_whitening = zca_whitening,
                                shear_range= shear_range, # Shear angle in counter-clockwise direction in degrees
                                zoom_range= zoom_range, #Range for random zoom
                                rotation_range= rotation_range, #Degree range for random rotations.
                                rescale= rescale_1/rescale_2) # rescaling factor 

    #define directory
    directory= os.path.join(os.getcwd(),"data","archive","images")

    # training data
    train_ds = datagen.flow_from_dataframe(
                        dataframe= train_df,
                        directory = directory,
                        x_col="image",
                        y_col="label",
                        batch_size=batch_size,
                        seed=666,
                        shuffle=True,
                        class_mode= class_mode,
                        target_size=target_size)


    # validation data
    val_ds =datagen.flow_from_dataframe(
                        dataframe=val_df,
                        directory = directory,
                        x_col="image",
                        y_col="label",
                        batch_size=batch_size,
                        seed=666,
                        shuffle=True,
                        class_mode= class_mode,
                        target_size=target_size)


    test_datagen=ImageDataGenerator(rescale=rescale_1/rescale_2)

    # test data
    test_ds =test_datagen.flow_from_dataframe(
                        dataframe=test_df,
                        directory = directory,
                        x_col="image",
                        y_col="label",
                        batch_size=batch_size,
                        seed=666,
                        shuffle=False,
                        class_mode= class_mode,
                        target_size=target_size)
    
    return(train_ds, val_ds, test_ds)


############## LOAD MODEL ################

def loading_model(pooling, input_shape,nodes_layer_1, nodes_layer_2, activation_hidden_layer, activation_output_layer, initial_learning_rate, decay_steps, decay_rate, loss): 
    # load the pretrained VGG16 model without classifier layers
    model = VGG16(include_top=False, 
                pooling= pooling, 
                input_shape= input_shape)


    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False #resetting 
    # we only wanna update the classification layer in the end,
    # so now we "freeze" all weigths in the feature extraction part and make them "untrainable"


    ########## adding classification layers #########

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1) #normalize the feature weights 
    # 1st layer
    class1 = Dense(nodes_layer_1, 
                activation=activation_hidden_layer)(bn)
    # 2nd layer               
    class2 = Dense(nodes_layer_2, 
                activation=activation_hidden_layer)(class1)

    # output layer    
    output = Dense(7, #7 lables
                activation=activation_output_layer)(class2) 

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
                loss=loss,
                metrics=['accuracy'])
    
    return(model)



############## FIT & TRAIN #################

def fitting_model(model, train_ds, n_epochs, val_ds, monitor, patience, restore_best_weights, batch_size):
    #model
    skin_cancer_classifier = model.fit(train_ds,
                        steps_per_epoch= train_ds.samples // batch_size,
                        epochs = n_epochs,
                        validation_data=train_ds,
                        validation_steps= val_ds.samples // batch_size,
                        batch_size = batch_size,
                        verbose = 1,
                        callbacks=[EarlyStopping(monitor=monitor, patience=patience, restore_best_weights =  restore_best_weights)]
                        )

    return skin_cancer_classifier, model


############ EVALUATION #####################
def model_evaluation(skin_cancer_classifier, n_epochs): 
    hf.plot_history_ubal(skin_cancer_classifier, n_epochs)
    return()

################### MODEL PREDICTIONS ########################

def model_predictions(model, test_ds, batch_size):        
    predictions = model.predict(test_ds, # X_test
                                batch_size=batch_size)
    return predictions 

################### SAVE RESULTS ########################

def save_results(test_ds, predictions):
    # Make classification report
    report=(classification_report(test_ds.classes, # y_test 
                                                predictions.argmax(axis=1),
                                                target_names=test_ds.class_indices.keys())) #labels

    print(report)
    # Define outpath for classification report
    outpath_report = os.path.join(os.getcwd(), "out", "classification_report_unbalanced.txt")

    # Save the  classification report
    file = open(outpath_report, "w")
    file.write(report)
    file.close()

    print( "Saving the skin cancer classification report in the folder ´out´")

    return()


#################### MAIN FUNCTION #######################
def main():
    args = input_parse()

    #image data generator functions
    train_ds, val_ds, test_ds = data_augmentation(train_df, test_df, val_df, args.horizontal_flip, args.vertical_flip, args. zca_whitening, args.shear_range, args.zoom_range, args.rotation_range, args.rescale_1, args.rescale_2, args.batch_size, args.class_mode, tuple(args.target_size))

    #model functions
    model = loading_model(args.pooling, tuple(args.input_shape), args.nodes_layer_1, args.nodes_layer_2, args.activation_hidden_layer, args.activation_output_layer, args.initial_learning_rate, args.decay_steps, args.decay_rate, args.loss)
    skin_cancer_classifier, model = fitting_model(model, train_ds, args.n_epochs, val_ds, args.monitor, args.patience, args.restore_best_weights, args.batch_size)
    model_evaluation(skin_cancer_classifier, args.n_epochs)
    predictions = model_predictions(model, test_ds, args.batch_size)
    save_results(test_ds, predictions)

if __name__ == '__main__':
    main()
