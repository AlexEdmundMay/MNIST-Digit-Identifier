# -*- coding: utf-8 -*-
"""
############################################################################
Created on Fri Mar 12 15:44:56 2021

############################################################################
Author: Alexander May
Email: alexander.may-2@student.manchester.ac.uk

############################################################################
MNIST Digit Identification using Tensor Flow

############################################################################
Identifies Hand-Written Digits Correctly Approximately 97% of the time
        (Depending on which data is selected for training)

############################################################################
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def loadData():
    """
    -Loads data from the MNIST dataset
    -Normalises X data for slight accuracy improvement
    -Returns Data
    """
    #Load Data
    (X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
  
    #Normalise Data
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    
    #Return Data
    return (X_train,y_train),(X_test,y_test)

def createModel():
    """
    Creates and returns a Sequential Model which:
        -Flattens 28x28 pixel image into 1D Array
        -Uses Two Dense Hidden Layers (Rectified Linear Activation)
        -Output Layer Uses One-Hot Coding i.e. 3 --> [0,0,0,1,0,0,0,0,0,0]
            and a softmax activation to preserve order of probabilities
    """
    
    model = tf.keras.Sequential()   #Creates Sequential Model
    
    model.add(tf.keras.layers.Flatten())   #Flattens Image
    
    #Deanse Hidden Layers
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    
    #Output Layer
    model.add(tf.keras.layers.Dense(10, activation =tf.nn.softmax))
    
    return model

def testModel(model, X_test):
    """
    Makes digit prediction given a trained model, model, and test data, X_test
    """
    #np.argmax() converts from One-Hot Coding to Int representation
    predictions = np.argmax(model.predict(X_test), axis=1)
    return predictions

def main():
    """
    Loads Data
    Creates and Trains the Model
    Predicts Labels of Test Data
    Plots Some Successful Guesses
    """
    #Load Data
    (X_train,y_train),(X_test,y_test) = loadData()
    
    #Create and Compile Model
    model = createModel()
    model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    
    #Fit Model to Training Data
    model.fit(X_train,y_train,epochs=5)
    
    #Test Model using Test Data
    predictions = testModel(model,X_test)
    
    #Separates Successful and Failed Predictions
    failed_indices = []
    successful_indices = []
    for i in range(len(predictions)):
        if predictions[i] != y_test[i]:
            failed_indices.append(i)
        else:
            successful_indices.append(i)
    
    test_accuracy = len(successful_indices)/len(predictions)
    print("\n\nTest Data Accuracy =",test_accuracy)
    
    plt.close("all")    #Closes all Figures Currently Open
    
    #To demonstrate it works to user, Show 10 images with the computer's guess
    for index in successful_indices[0:10]:
        #creates new plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Comp Guessed: "+str(predictions[index]))
        
        #Show Image
        ax.imshow(X_test[index],cmap=plt.cm.binary)
    
    plt.show()   #Show Images


#Runs main() Function
if __name__ == '__main__':
    main()
