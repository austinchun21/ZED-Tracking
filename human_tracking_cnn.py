"""
Human Tracking CNN

Based on: 
Author: Austin Chun
Email:  austinchun21@gmail.com
Date:   June 2018

CNN implemented in Keras (TensorFlow) to track humans using ZED stereo camera.
"""
from __future__ import print_function

# Keras 
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

# Custom Imaging classes
# from generate_data import InitialImaging, PredictImaging, UpdateImaging
from initial_data_gen import InitialDataGen
from predict_data_gen import PredictDataGen
from update_data_gen import UpdateDataGen

# General libraries
import cv2
import numpy as np
import math
from time import time
import matplotlib.pyplot as plt

# ROS
import sys
import rospy
from sensor_msgs.msg import Image # , Float32
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError

# Extras
import argparse
import select

# Parameters
EPOCHS = 50
BATCH_SIZE = 80

UPDATE_EPOCHS = 2
UPDATE_BATCH_SIZE = 80

# Save directory
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'human_tracking_trained_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

# dataDir = 'data/lab_and_seminar/'
dataDir = 'data/corridor_corners/'

# Define FONT style
FONT = cv2.FONT_HERSHEY_SIMPLEX


def coords2centroid(coords):
    """ Convert [x,y,w,h] to centroid coordinates (xc,yc) """
    return coords[0]+coords[2]//2, coords[1]+coords[3]//2

def enter():
    """ Non-blocking wait until ENTER"""
    return select.select([sys.stdin], [], [], 0)[0]

class HumanTracker_CNN():
    """
    Main class that implements the Human Tracking CNN.

    Inputs are RGB and Depth images. Outputs the estimated centroid
    of the target.

    CNN doesn't require any pretraining, instead trains/updates online.
    """

    def __init__(self, args, SHOW_INIT_PATCHES=False, SHOW_PATCH_SEARCH=False,  SHOW_BEST_PATCH=False, SHOW_UPDATE_PATCHES=False):
        """
        Initialization of Human Tracking CNN

        Kwargs:
            SHOW_INIT_PATCHES:  Show Init patch training set
            SHOW_PATCH_SEARCH:  Show patches searched through before classification step

        """

        self.DEBUG = args.DEBUG
        self.REALTIME = args.REALTIME

        if(self.REALTIME):
            self.IMG_W, self.IMG_H, self.IMG_C = 1280, 720, 4
            self.INIT_W, self.INIT_H = 200, 700
            self.FOV = 90.0 # 90 deg Field of View for ZED camera
        else:
            self.IMG_W, self.IMG_H, self.IMG_C = 672, 376, 4
            self.INIT_W, self.INIT_H = 100, 350
            self.FOV = 90.0 # 90 deg Field of View for ZED camera
        self.CNN_SIZE = 28


        # Init centroid to middle
        self.centroid = (self.IMG_W//2, self.IMG_H//2)
        self.boxCoord = (self.IMG_W//2-self.INIT_W//2, self.IMG_H//2 - self.INIT_H//2, self.INIT_W, self.INIT_H) # (x,y,w,h)
        # self.depth = 1.25 *256/6.0 # Initialize to 1.0 m 
        self.depth = -1
        self.angle = 0
        self.CONF_THRESHOLD = 0.5 # Threshold for acquiring target

        self.count = 0
        self.framsesSinceLost = 0

        self.lostThresh = 5 # 10 frames, before considered lost

        # ROS ZED Camera
        if(self.REALTIME):
            # ROS Setup
            self.rosRate = rospy.Rate(10)
            self.GOT_ROS_DEPTH = False # Flag to indicate new frame
            self.GOT_ROS_BGR = False
            self.READING_IMAGE = False
            self.bridge = CvBridge()
            self.depth_sub = rospy.Subscriber("/zed/depth/depth_registered", Image, self._depth_cb)#, queue_size=2)
            self.left_sub  = rospy.Subscriber("/zed/left/image_rect_color",  Image, self._bgr_cb)#,   queue_size=2)
        
            self.depth_pub = rospy.Publisher("/CNN/depth", Float32, queue_size = 10)
            self.angle_pub = rospy.Publisher("/CNN/angle", Float32, queue_size = 10)


	# Dataset
        else:
            # Read in Ground Truth 
            f = open(dataDir+'GroundTruth.txt')
            # Nested list comp: loop every line in file, change each value to an int (tab separated)
            self.true_coords = np.array([[int(val) for val in line.split('\t')] for line in f.readlines()])
            # Delete first column
            self.true_coords = np.delete(self.true_coords, 0, 1)


        # Initialize Imaging class
        self.dims = [self.IMG_W, self.IMG_H, self.IMG_C, self.INIT_W, self.INIT_H, self.CNN_SIZE]
        self.PI = PredictDataGen(self.dims, SHOW_PATCH_SEARCH=SHOW_PATCH_SEARCH)

        # Initialize Update Image class
        self.UI = UpdateDataGen(self.dims, SHOW_UPDATE_PATCHES=SHOW_UPDATE_PATCHES)

        # Initialize CNN Model 
        self.model = self._baseModel()


    def _depth_cb(self, data):
        try:
            # Convert Image msg to cv2 (nparray)
            self.depthImg = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough") 
            # Convert float32 to uint8 (scale from 0-6 meters, to 0-255 greyscale)
            self.depthImg = (255./6 * self.depthImg).astype(np.uint8)

        except CvBridgeError as e:
            print(e)


    def _bgr_cb(self, data):
        try:
            # Convert Image msg to cv2 (nparray)
            self.bgrImg = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


    def _baseModel(self):
        """
        Constructs and compiles the layers of the CNN.

        Returns:
            model:  A Keras Sequential model (untrained)
        """

        # Create the Model structure
        model = Sequential()
        model.add(Conv2D( 32, (3,3), strides=1, padding='same', use_bias=True, activation='relu', input_shape=(self.CNN_SIZE,self.CNN_SIZE,self.IMG_C)))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

        model.add(Conv2D(64, (3,3), strides=1, padding='same', use_bias=True, activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # Initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=0.0) #1e-6)
        # Let's train the model using RMSprop
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        return model


    def trainCNN(self, index=0, epochs=10, batch_size=41, SHOW_INIT_PATCHES=False):
        """
        Trains the CNN on Initialization data (generated internally, specified by img index)

        Can use class_weights depending on balance of initial dataset.
        (ex: If using say 40 class-0 and 1 class-1, then class_weight={0:1. , 1:40. , 2:2. })
        (Not sure why 2 is needed)

        Kwargs:
            index:      Image ID to generate (or start generating) class-1 and class-0 patches
            epochs:     Number of epochs to run
            batch_size: Typically the size of the dataset (since dataset is small) 
        """

        self.index = index
        # Get image
        self._getImg()
        # Keep polling until valid depth set
        while(self.depth == -1):
            self._getTargetDepth()

        print("Init depth: %.3f" %self.depth)

        # Generate Training Data
        InitData = InitialDataGen(self.img, self.dims, SHOW_INIT_PATCHES=SHOW_INIT_PATCHES)
        (data, labels) = (InitData.patches, InitData.labels)

        # Train the model (use class_weight b/c unbalanced data)
        class_weight = {0: 1. , 1: 1, 2: 2.}
        self.model.fit(data, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight)

        # Evaluate model on Training set
        scores = self.model.evaluate(data, labels, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])


    def saveModel(self):
        """ Save model and weights"""
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)


    def predictCentroid(self, index, SHOW_PATCH_PREDICTIONS=False, SHOW_REG_PLOTS=True, SHOW_BEST_PATCH=False):
        """
        Predicts the Centroid using the trained CNN

        Generates possible patches from Predict Imager. Then classifies those 
        patches with trained CNN. Takes the maximum result patch and coordinates
        If result confident enough (>0.5), sets centroid, boxCoords, and depth. 

        Args:
            index:                  Image ID to extract patches to classify, and estimate centroid
        Kwargs:
            SHOW_PATCH_PREDICTIONS: Display patch with its prediction
            SHOW_REG_PLOTS:         Show regression plots, ex CNN prediction vs patch_width
            SHOW_BEST_PATCH:        Display best patch

        Return:
            Sets centroid, boxCoords, and depth if confident enough
        """ 
        self.count += 1

        self.index = index
        # Get image
        self._getImg()

        # Update last target location 
        self.PI.setLastPos(self.boxCoord, self.depth*255/6.0)

        # Input the image, and generate patches (along with coords)
        LOST = self.framsesSinceLost > self.lostThresh
        patches, coords = self.PI.generateData(self.img, LOST)

        if(LOST):
            print("\t TARGET LOST!")


        # Log all predictions and centDists
        if(self.DEBUG):
            predictions = np.empty((0,1), int)
            centDists   = np.empty((0,1), int)
            patchCoords = np.empty((0,4), int)

        # Truth Data
        if(not self.REALTIME):
            # Extract dimensions
            tx, ty, tw, th = self.true_coords[index]
            # True Centroid coordinates
            txc, tyc = coords2centroid([tx,ty,tw,th])

        # Draw ground truth box
        if(SHOW_PATCH_PREDICTIONS):
            origDrawOn = self.bgrImg.copy() # Copy for drawing on 
            # Add Truth box and centroid
            cv2.rectangle(origDrawOn, (tx,ty),(tx+tw,ty+th), color=(0,255,0),thickness=2)
            cv2.circle(origDrawOn, (tx+tw//2,ty+th//2), 2, color=(0,255,0), thickness=2)


        # Initialize parameter to maximize
        maxPredict = 0

        # Loop through patches
        for j in range(patches.shape[0]):
            # Get coordinates
            x,y,w,h = coords[j]
            # Classify the patch
            testImg = [patches[j,:,:,:]]
            prediction = self.model.predict([testImg])[0][0]

            if(prediction > maxPredict):
                maxPredict = prediction
                bestCoord  = [x,y,w,h] 

            if(self.DEBUG):     
                # Log data
                predictions = np.append(predictions, [[prediction]], axis=0)
                patchCoords = np.append(patchCoords, [coords[j]], axis=0)
                if(not self.REALTIME):
                    # Calculate Centroid Error
                    xc, yc = coords2centroid([x,y,w,h])
                    centDist = math.sqrt(math.pow(txc-xc,2) + math.pow(tyc-yc,2)) 
                    centDists = np.append(centDists, [[centDist]], axis=0)


            # Draw patch with prediction too
            if(SHOW_PATCH_PREDICTIONS and not self.REALTIME):
                drawOn = origDrawOn.copy() # Create copy (erase old image)
                cv2.rectangle(drawOn, (x,y),(x+w,y+h), color=(255,0,0), thickness=2) # box
                cv2.circle(drawOn, (x+w//2,y+h//2), 2, color=(255,0,0), thickness=2)   # center
                cv2.putText(drawOn,'Pred: '+str(round(prediction,2)),(10,50), FONT, 1,(0,0,0),2)
                if(not self.REALTIME):
                    cv2.putText(drawOn,'Dist: '+str(round(centDist,2)),(10,100), FONT, 1,(0,0,0),2)
                cv2.imshow('Truth vs. Patch', drawOn[:,:,0:3])
                cv2.waitKey(100) # Wait for 100 ms

        # The maximum confidence
        self.centConfidence = maxPredict
        # only update if confidence is high enough
        if(self.centConfidence > self.CONF_THRESHOLD):
            self.centroid = coords2centroid(bestCoord)
            self.boxCoord = bestCoord
            self._getTargetDepth()
            self._getTargetAngle()
            self.framsesSinceLost = 0
        else:
            self.framsesSinceLost += 1

        # Plot some trends
        if(SHOW_REG_PLOTS):
            if(not self.REALTIME):
                # Plot Predict vs Centroid Diff
                fig = plt.figure(1)
                plt.clf()
                fig.patch.set_facecolor('white')
                plt.plot(centDists, predictions, 'bo')
                # plt.axis([0,70,0,1])
                plt.ylim(0,1)
                plt.grid(True)
                plt.title('CNN Prediciton vs Centroid Diff')
                plt.xlabel('Centroid Difference (pix)')
                plt.ylabel('CNN Prediction')

            # Plot Predict vs Patch params
            fig = plt.figure(2)
            plt.clf()
            fig.patch.set_facecolor('white')

            plt.subplot(221)
            plt.plot(patchCoords[:,0], predictions, 'bo')
            plt.xlim(186,386)
            plt.grid(True)
            plt.xlabel('Patch X Coord (pix)')
            plt.ylabel('CNN Prediction')
           
            plt.subplot(222)
            plt.plot(patchCoords[:,1], predictions, 'bo')
            plt.xlim(0,60)
            plt.grid(True)
            plt.xlabel('Patch Y Coord (pix)')
            plt.ylabel('CNN Prediction')

            plt.subplot(223)
            plt.plot(patchCoords[:,2], predictions, 'bo')
            plt.xlim(50,170)
            plt.grid(True)
            plt.xlabel('Patch Width (pix)')
            plt.ylabel('CNN Prediction')

            plt.subplot(224)
            plt.plot(patchCoords[:,3], predictions, 'bo')
            plt.xlim(270,376)
            plt.grid(True)
            plt.xlabel('Patch Height (pix)')
            plt.ylabel('CNN Prediction')

            plt.ion()
            plt.show()
        # Show best patch
        if(SHOW_BEST_PATCH):
            x,y,w,h = bestCoord
            bestPatch = self.bgrImg.copy()
            cv2.rectangle(bestPatch, (x,y),(x+w,y+h), color=(255,0,0), thickness=2)
            cv2.circle(bestPatch, (x+w//2,y+h//2),2,(255,0,0),thickness=2)
            cv2.rectangle(bestPatch, (0,0),(200,90), color=(255,255,255), thickness=-1)
            cv2.putText(bestPatch,'Frame: '+str(index),(10,15), FONT, 0.5,(0,0,0),2)
            cv2.putText(bestPatch,'Pred: '+str(round(self.centConfidence,3)),(10,35), FONT, 0.5,(0,0,0),2)
            cv2.putText(bestPatch,'Depth (m): '+str(round(self.depth,2)),(10,55), FONT, 0.5,(0,0,0),2)
            cv2.putText(bestPatch,'Angle (deg): '+str(round(self.angle,2)),(10,75), FONT, 0.5,(0,0,0),2)
            if(not self.REALTIME):
                cv2.rectangle(bestPatch, (tx,ty),(tx+tw,ty+th), color=(0,255,0),thickness=2)
                cv2.circle(bestPatch, (tx+tw//2,ty+th//2), 2, color=(0,255,0), thickness=2)
            cv2.imshow('bestPatch',bestPatch)
            cv2.waitKey(1)


    def updateCNN(self, epochs=1, batch_size=80):
        """
        Generates new training set, and fits keras model to it

        Kwargs:
            epochs:     number of epochs
            batch_size: update batch size (typically 80, can be different from init batch_size)
        Returns:
            Fits the Keras model to the new training set
        """

        # Generate patches and labels for training
        (patches, labels) = self.UI.generateUpdateData(self.img, self.boxCoord)

        # Fit new data
        class_weight = {0: 1. , 1: 1., 2: 1.}
        self.model.fit(patches, labels, epochs=UPDATE_EPOCHS, batch_size=UPDATE_BATCH_SIZE, class_weight=class_weight)


    def _getTargetDepth(self):
        r = 20  # Radius of local average
        xc, yc = self.centroid
        local = self.depthImg[ (yc-r):(yc+r) , (xc-r):(xc+r)]

        # Only average pixels within reasonable range
        if (self.depth == -1): # Startup
            depth = np.mean(local)
        else:
            under = local > self.depth*255/6.0-self.PI.ALPHA
            over  = local < self.depth*255/6.0+self.PI.ALPHA
            joint = np.logical_and(under,over)
            depth = np.mean(local[joint])

        if(np.isfinite(depth)):
            self.depth = depth *6.0/255 # in meters
            if(self.REALTIME):
                # Publish to ROSTOPIC
                self.depth_pub.publish(self.depth)

    def _getTargetAngle(self):
        #   angle = (xc - middle) / width * FOV
        self.angle = ((self.boxCoord[0]+self.boxCoord[2]/2)-self.IMG_W/2) * self.FOV / (1.0*self.IMG_W)  # degress
        if(self.REALTIME):
            # Publish to ROSTOPIC
            self.angle_pub.publish(self.angle)

    def _getImg(self):
        """
        Compiles the bgr and depth images into RGBSD (self.img)

        If running realtime with ROS, images loaded in callback.
        Otherwise using dataset, then load images from files

        Args:
            index:  used for indexing images if needed
        Returns:
            Sets self.img to RGBSD image (4 channel nparray)
        """

        if(not self.REALTIME):
            # Load image file
            self.bgrImg = cv2.imread(dataDir+'left/left'+ str(self.index).zfill(8) + '.jpg')
            self.depthImg   = cv2.imread(dataDir+'depth/depth'+str(self.index).zfill(8) + '.jpg', 0) # Read as greyscale (just in case)
        
        # else, images set in ROS callback 

        # Combine into RGBSD img
        self.depthImg = np.reshape(self.depthImg, (self.IMG_H,self.IMG_W,1))
        self.img = np.concatenate((self.bgrImg, self.depthImg), axis=2)



def main():

    # Argument parser
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('-d', action="store_true", dest='DEBUG', default=False)
    parser.add_argument('-rt', action="store_true", dest='REALTIME', default=False)
    args = parser.parse_args()
    print(" DEBUG: %r" %args.DEBUG)
    print(" REALTIME: %r" %args.REALTIME)


    # Flags and stuff
    SHOW_INIT_PATCHES           = True
    SHOW_UPDATE_PATCHES         = False
    SHOW_PATCH_PREDICTIONS      = False
    SHOW_REG_PLOTS              = False
    SHOW_BEST_PATCH             = True

    # Initialize ROS node
    if(args.REALTIME):
        rospy.init_node('human_tracker_CNN', anonymous=True)
        print("Started ROS node")
        r = rospy.Rate(10) # 10 Hz

    # Create CNN Class (And trains)
    CNN = HumanTracker_CNN(args, SHOW_INIT_PATCHES=SHOW_INIT_PATCHES,
                            SHOW_UPDATE_PATCHES=SHOW_UPDATE_PATCHES)

    if(CNN.REALTIME):
        print("Align yourself inside the box on the scren, then press ENTER...")
        while(not enter()):
            cv2.rectangle(CNN.bgrImg, (CNN.IMG_W//2-CNN.INIT_W//2, CNN.IMG_H//2-CNN.INIT_H//2), 
                                          (CNN.IMG_W//2+CNN.INIT_W//2, CNN.IMG_H//2+CNN.INIT_H//2),
                                          color=(0,200,0), thickness=2)
            cv2.circle(CNN.bgrImg, (CNN.IMG_W//2, CNN.IMG_H//2), 3, color=(0,200,0), thickness=2)
            cv2.imshow("Starting Box", CNN.bgrImg)
            cv2.waitKey(1)

    print("Starting the CNN...")

    # Train model on training set (IMG)
    CNN.trainCNN(index=0, epochs=EPOCHS, batch_size=BATCH_SIZE, SHOW_INIT_PATCHES=SHOW_INIT_PATCHES)

    if(CNN.DEBUG):
        # Initialize arrays for logging
        centErrs  = np.empty((0,1), int)
        centConf  = np.empty((0,1), int)
        centroids = np.empty((0,2),int)
        boxCoords = np.empty((0,4),int)


    # Predict an image    
    for j in range(1,1200):
        try:
            # Predict centroid
            CNN.predictCentroid(j, SHOW_PATCH_PREDICTIONS=SHOW_PATCH_PREDICTIONS, 
                                SHOW_REG_PLOTS=SHOW_REG_PLOTS, SHOW_BEST_PATCH=SHOW_BEST_PATCH)

            # if(j==1):
            #     print("Press ENTER to continue (on image if present)...")
            #     cv2.waitKey(0)

            # Print stuff
	    if(j%20 == 0):
            	print("------ Img %s -------" %(str(j).zfill(4)))
            if(CNN.DEBUG):
                print("Centroid: ",CNN.centroid)
                print("Depth: ", CNN.depth*6.0/255)
                print("Confidence:",CNN.centConfidence)
                print("Box Coord:",CNN.boxCoord)
                # Log data into arrays
                centConf = np.append(centConf, np.array([[CNN.centConfidence]]), axis=0)
                centroids = np.append(centroids, np.array([CNN.centroid]), axis=0)
                boxCoords = np.append(boxCoords, np.array([CNN.boxCoord]), axis=0)
                
                if(not CNN.REALTIME):
                    # Calculate Centroid Error
                    xc, yc = CNN.centroid
                    tx, ty = coords2centroid(CNN.true_coords[j])
                    centErr = math.sqrt(math.pow(xc-tx,2) + math.pow(yc-ty,2))
                    centErrs = np.append(centErrs, np.array([[centErr]]), axis=0)
                    print("True Coord:",CNN.true_coords[j])
                    print("Centroid Err: ", centErr)
            
            # Aim for 10 Hz
            if(CNN.REALTIME):
                r.sleep()
        except KeyboardInterrupt:
            print("Shutting down")
            break

    if(CNN.DEBUG):
        if(not CNN.REALTIME):
            fig = plt.figure(10)
            fig.patch.set_facecolor('white')
            plt.plot(centErrs)
            plt.grid(True)
            plt.title('CNN Error')
            plt.xlabel('Image Index')
            plt.ylabel('Centroid Err')

        fig = plt.figure(11)
        fig.patch.set_facecolor('white')
        plt.plot(centConf)
        plt.grid(True)
        plt.title('CNN Confidence')
        plt.xlabel('Image Index')
        plt.ylabel('Centroid Confidence')

        fig = plt.figure(12)
        fig.patch.set_facecolor('white')
        plt.plot(centroids[:,0])
        plt.hold(True)
        plt.plot(centroids[:,1])
        plt.plot(boxCoords[:,2])
        plt.plot(boxCoords[:,3])
        plt.grid(True)
        plt.title('Centroid Coordinates')
        plt.xlabel('Image Index')
        plt.ylabel('Coordinate (pix)')
        plt.legend(['Xc','Yc','W','H'])

        plt.show()

    # Close all windows
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
