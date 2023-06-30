# Custom Pylint rules for the file
# pylint: disable=E0401 W0719 R0913 C0301 W0718 E1101
# E0401:import-error
# W0719:broad-exception-raised
# R0913:too-many-arguments
# C0301:line-too-long
# W0718:broad-exception-caught
# E1101:no-member

import tensorflow as tf
import os
import cv2
from sklearn.utils import shuffle
import numpy as np
from random import randint
import matplotlib.pyplot as plot
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_images(directory, seed):
    """
    Loads the dataset.
    """
    images = []
    labels = []  
    label = 0
    
    for lab in os.listdir(directory):
        if lab == 'glacier':
            label = 2
        elif lab == 'sea':
            label = 4
        elif lab == 'buildings':
            label = 0
        elif lab == 'forest':
            label = 1
        elif lab == 'street':
            label = 5
        elif lab == 'mountain':
            label = 3
        
        for image_file in os.listdir(directory + lab): # Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory+lab+r'/'+image_file) # Reading the image (OpenCV)
            images.append(image)
            labels.append(label)
    
    return shuffle(images,labels,random_state=seed) # Shuffle the dataset you just prepared.

def resize_images(list_images, max_size):
    """
    Resize list of loaded images.
    """
    images = []
    for image in list_images:
        res_image = cv2.resize(image,(max_size, max_size))
        images.append(res_image)
    return images
    
def get_classlabel(class_code):
    """
    TODO: docstring
    """
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}
    
    return labels[class_code]

def viz_random_images(images, labels):
    """
    Visualize some random images from the dataset.
    """
    fig, ax = plot.subplots(5,5) 
    fig.subplots_adjust(0,0,3,3)
    for i in range(0,5,1):
        for j in range(0,5,1):
            rnd_number = randint(0,len(images))
            ax[i,j].imshow(images[rnd_number])
            ax[i,j].set_title(get_classlabel(labels[rnd_number]))
            ax[i,j].axis('off')

def plot_train_history(model):
    """
    Plots a trained model val_accuracy and val_loss history.
    """
    history_df = pd.DataFrame(model.history)

    fig_acc = px.line(history_df, y=['accuracy', 'val_accuracy'], labels={'index': 'Epoch', 'value': 'Accuracy'})
    fig_acc.update_layout(title='Model Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
    fig_acc.show()

    fig_loss = px.line(history_df, y=['loss', 'val_loss'], labels={'index': 'Epoch', 'value': 'Loss'})
    fig_loss.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
    fig_loss.show()

def predict_random_image(model, test_imgs, test_labels):
    """
    Does a random prediction for a given model.
    """
    # Select a random index from the test set
    random_index = np.random.randint(0, len(test_imgs))
    image = test_imgs[random_index]
    label = np.argmax(test_labels[random_index])
    image = np.expand_dims(image, axis=0)

    # Perform prediction using the model
    predictions = model.predict(image)
    predicted_category = np.argmax(predictions[0])
    predicted_label = get_classlabel(predicted_category)

    # Display the predicted category, label, and image
    plt.imshow(image.squeeze())
    if predicted_category == label:
        print("✅ Correct!")
    else:
        print("🔴 Wrong!")
    print(f"Predicted Category: {get_classlabel(predicted_category)}")
    print(f"Real Category: {get_classlabel(label)}")
