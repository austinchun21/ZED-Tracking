"""
CNN.py

    Implement the CNN class. Uses Keras TensorFlow.
    Used in ZED Human Tracking algorithm. YOLO will produce bounding boxes, 
    this CNN will distinguish target from others

    CNN Architecture 
        Input Shape = 64x64x3
        Conv2D      16 filters
        MaxPool
        Conv2D      32 filters
        MaxPool
        Flatten
        Dense       128 neurons
        Dropout
        Dense       2 classes

    Model training inspired by: https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

    Author: Austin Chun  <austinchun21@gmail.com>
    Date:   August 11th, 2018

"""

# Keras 
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import backend as K
from imutils import paths
import random

import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage import img_as_float

# General libraries
import cv2
import numpy as np
import math
from time import time
import matplotlib.pyplot as plt
import glob


# Hyper Parameters
EPOCHS = 20
BATCH_SIZE = 50
INIT_LR = 1e-3

# Save directory
model_name = 'ZED_target_CNN.h5'

TRAIN_DIR = 'data/train/'                   # Should have two folders, 'target/' and 'not_target/'
TEST_DIR = 'data/test/*'                    # All images in this file
TRAINING_PLOT_NAME = 'training_plot.png'    # Where to save training plot


class TargetCNN():


    def __init__(self):

        # CNN takes a 32x32x3 numpy array
        self.CNN_SIZE = 64
        self.CHANS = 3

        # self.model = self._baseModel()
        self.model = self._baseModel()


    def _baseModel(self):
        """
        Constructs and compiles the layers of the CNN.

        Returns:
            model:  A Keras Sequential model (untrained)
        """

        # Create the Model structure
        model = Sequential()
        inputShape = (self.CNN_SIZE, self.CNN_SIZE, self.CHANS)
        model.add(Conv2D( 16, (3,3), strides=1, padding='same', use_bias=True, activation='relu', 
                          input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

        model.add(Conv2D(32, (3,3), strides=1, padding='same', use_bias=True, activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2, activation='softmax'))

        # Initiate ADAM optimizer
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        return model

    def train(self, PLOT_LEARNING=False, epochs=EPOCHS, trainDir=TRAIN_DIR):

        print("[INFO] loading images...")
        data = []
        labels = []

        # grab the image paths and randomly shuffle them
        imagePaths = sorted(list(paths.list_images(trainDir)))
        random.seed(42)
        random.shuffle(imagePaths)

        # loop over the input images
        for imagePath in imagePaths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (self.CNN_SIZE, self.CNN_SIZE))
            image = img_to_array(image)
            data.append(image)

            # extract the class label from the image path and update the
            # labels list
            label = imagePath.split(os.path.sep)[-2]
            label = 1 if label == "target" else 0
            labels.append(label)

        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)           

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (trainX, testX, trainY, testY) = train_test_split(data,
            labels, test_size=0.25, random_state=42)

        # convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=2)
        testY = to_categorical(testY, num_classes=2)

        # construct the image generator for data augmentation
        aug = ImageDataGenerator(
                    rotation_range=30, 
                    width_shift_range=0.1,
                    height_shift_range=0.1, 
                    shear_range=0.2, 
                    zoom_range=0.2,
                    horizontal_flip=True, 
                    fill_mode="nearest")
        

        # train the network
        print("[INFO] training network...")
        H = self.model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
            epochs=epochs, verbose=1)

        if(PLOT_LEARNING):
            # plot the training loss and accuracy
            plt.style.use("ggplot")
            plt.figure()
            N = epochs
            plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
            plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
            plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy on Target/Not Target")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            # plt.savefig(TRAINING_PLOT_NAME)
            plt.show()



    def predict(self, patch):
        """
        Use CNN to predict if the given patch is the target (or some other person).
        Resizes the patch (to fit CNN), and demean 
        Inputs:
            patch:          nparray     RGB image (3 chan), cropped portion of video frame
        Return:
            prediction:     nparray     CNN prediciton (confidence) in classification
        """
        
        # Preprocess (resize)
        small_patch = cv2.resize(patch, (self.CNN_SIZE, self.CNN_SIZE)) # Resize patch to be CNN size
        small_patch = small_patch.astype("float") / 255.0
        small_patch = img_to_array(small_patch)
        small_patch = np.expand_dims(small_patch, axis=0)

        # Classify!
        (notTarget, target) = self.model.predict(small_patch)[0]
        
        # build the label
        label = "Target" if target > notTarget else "Not Target"
        prob = target if target > notTarget else notTarget
        pred = prob if label == "Target" else (1-prob)
        # print_label = "{}: {:.2f}%".format(label, prob * 100)

        return label, prob, pred


    def saveModel(self):
        """ Save model and weights"""
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)




def main():

    # Initialize
    CNN = TargetCNN()

    #############################
    ### Test Training for CNN ###
    #############################
    for i in range(1):
        CNN.train(PLOT_LEARNING=True)

        # Counts
        right = 0
        count = 0.0

        # Intialize arrays
        targets = np.zeros(0)       # Keep track of target classifications
        others = np.zero(0)         # Other people classifications
        not_targets = np.zeros(0)   # other not_target classifications (from pedestrian database)

        targ_count = 0
        not_targ_count = 0

        # image_list = []
        for filename in glob.glob(TEST_DIR):
            im = cv2.imread(filename)
            label, prob, pred = CNN.predict(im)

            count += 1

            if('Target' in filename):
                targets = np.append(targets, prob)
                if(label == 'Target'):
                    right += 1
            else:
                not_targets = np.append(not_targets, 1-prob)
                if(label == 'Not Target'):
                    right += 1

        print(" Acc: %.3f"%(right*1.0/count))

        plt.figure()
        nbins = 10
        plt.subplot(211)
        plt.hist(targets, nbins)
        plt.xlim(0,1)
        plt.subplot(212)
        plt.hist(not_targets, nbins)
        plt.xlim(0,1)

        plt.show()

    # CNN.saveModel()

if __name__ == '__main__':
    main()

