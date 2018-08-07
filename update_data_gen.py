"""
update_data_gen.py

    Generate update data to retrain/update CNN.

    Author: Austin Chun
    Date:   Aug 2018

"""

from PIL import Image
import numpy as np
import cv2
import random
from random import randint
import math
import time
from skimage.transform import resize

# Random Seed for reproducibility
random.seed(21)
np.random.seed(21)

dataDir = 'data/corridor_corners/'
# dataDir = 'data/lab_and_seminar/'


class UpdateDataGen():
    """
    Class implements data generation for updating/retraining the CNN.

    After Initialization, and Prediction, if the prediction returns a 
    significant confidence for a class-1 patch, that patch is treated 
    as truth, and added to a pool of class-1 patches. Then 40 class-1 
    patches are selected based on a Poission distribution to construct
    the class-1 patch training set. The class-0 training set is taken
    from the current image, around the class-1 patch. 
    """

    def __init__ (self, dims, SHOW_UPDATE_PATCHES=False):
        """
        Initialization of Update Imaging class.

        Kwargs:
            SHOW_UPDATE_PATCHES:    Display the update training set
        """

        self.SHOW_UPDATE_PATCHES = SHOW_UPDATE_PATCHES

        self.IMG_W, self.IMG_H, self.IMG_C, _ , _ , self.CNN_SIZE = dims

        # Coordinates of last class1 patch (should be overridden before use)
        self.class1coords = [-1, -1, -1, -1]
        self.x1, self.y1, self.box_w, self.box_h = self.class1coords
        self.x2, self.y2 = self.x1 + self.class1coords[2], self.y1 + self.class1coords[3] 

        self.xc, self.yc = (self.x1+self.x2)//2, (self.y1+self.y2)//2


        self.queueSize = 15
        # self.queueSize = 50
        self.poissonPMF = self._generatePoisson(self.queueSize) # Generate Poisson dist for class-1 queue
        self.desNum0Patches = 40 # How many patches to use

        # Class 1 Update pool
        self.class1pool    = np.empty((0, self.CNN_SIZE, self.CNN_SIZE, self.IMG_C), int)
        self.class1poolCoords = np.empty((0,4), int) # Coords of patches

        # DEBUG: save all patches
        self.class1visuals = np.empty((0,self.IMG_W,self.IMG_H, self.IMG_C), int)

        # Sampled patches
        self.class1patches = np.empty((self.desNum0Patches, self.CNN_SIZE, self.CNN_SIZE, self.IMG_C), int)
        self.class0patches = np.empty((self.desNum0Patches, self.CNN_SIZE, self.CNN_SIZE, self.IMG_C), int)

    def generateUpdateData(self, img, coords):
        """
        Generates training set for update step.

        Args:
            img:    nparray of image (full dimensions 672x376x4)
            coords: nparray [1x4] of last target coords [x,y,w,h]
        Returns:
            nparray of patches, and nparray of labels
        """
        # Take in input img
        self.img = img


        if(self.SHOW_UPDATE_PATCHES):
            self.visual = self.img.copy()

        # Update last accurate box
        self._updateBoxCoords(coords)

        # Grab class0 pool
        self._grabClass0Patches3()

        # Update class1 pool with new patch
        self.updateClass1Pool(img, coords)

        # grab class1 pool
        self._grabClass1Patches()

        self.curQueueSize = self.class1pool.shape[0] 

        patches = np.append(self.class1patches, self.class0patches, axis=0)
        labels  = np.append(np.ones((self.class1patches.shape[0],1)),np.zeros((self.desNum0Patches,1)), axis=0)

        if(self.SHOW_UPDATE_PATCHES):
            cv2.imshow("Update Patches",self.visual)
            cv2.waitKey(0)

        return patches, labels

    def updateClass1Pool(self, img, coords):
        """
        Adds recent class1 patch to queue

        Args:
            newPatch: nparray of new patch (28x28x4)
        Returns:
            updates self.class1pool
        """

        self.img = img
        self._updateBoxCoords(coords)

        # Extract and resize patch
        newPatch = resize(self.img[self.y1:self.y2, self.x1:self.x2], (self.CNN_SIZE, self.CNN_SIZE) )

        # Still filling up queue
        if(self.class1pool.shape[0] < self.queueSize):
            self.class1pool = np.append(self.class1pool, [newPatch], axis=0)
        # Full queue, need to add/pop
        else:
            self.class1pool[1:, :,:,:] = self.class1pool[:-1, :,:,:] # Shift everything right
            self.class1pool[0,  :,:,:] = newPatch # Load new first index


    def _updateBoxCoords(self, coords):
        """
        Updates box coords locally (just a internal helper function for setting class variables)

        Args:
            coords: nparray of target coords [x,y,w,h]
        Returns:
            Sets class variables
        """
        # Load new target coords
        self.class1coords = coords
        self.x1, self.y1, self.box_w, self.box_h = self.class1coords
        # Calculate some helpful values
        self.x2, self.y2 = self.x1 + self.box_w, self.y1 + self.box_h 
        self.xc, self.yc = (self.x1+self.x2)//2, (self.y1+self.y2)//2

    def _grabClass0Patches3(self):
        # Use normal distribution, with empty space in the center of target
        
        # Center coordinates (assumed to be true centroid)
        txc, tyc = self.xc, self.yc
        
        stdX = 60 * self.IMG_W/672.0
        stdY = 20 * self.IMG_H/376.0

        # Set mean and std for varying bounding box sizes
        meanWidth, meanHeight = self.box_w, self.box_h
        stdWidth = 20  * self.IMG_W/672.0 # Adjustable std dev (pixels)
        stdHeight = 15 * self.IMG_H/376.0

        radiusThreshold = 60 * self.IMG_W/672.0

        for count in range(self.desNum0Patches):
            centDist, bw, bh = 0, 0, 0

            # Loop until found a valid patch (far enough away from centroid, and positive width/height)
            while(not (centDist > radiusThreshold and bw > 0 and bh > 0 )):
                bx = int(np.random.normal(txc-meanWidth/2, stdX))
                by = int(np.random.normal(tyc-meanHeight/2, stdY))
                
                # Make sure non-negative
                if(bx < 0):
                    bx = 0
                if(by < 0):
                    by = 0

                # Use random box sizes
                bw = int(np.random.normal(meanWidth, stdWidth))
                bh = int(np.random.normal(meanHeight, stdHeight))
                
                # Right box edge past img edge
                if( bx+bw >= self.IMG_W):
                    bw = self.IMG_W - bx -1
                # Bottom box edge past img edge
                if( by+bh >= self.IMG_H):
                    bh = self.IMG_H - by -1

                # Calculate centroid
                xc, yc = bx+bw//2, by+bh//2
                centDist = math.sqrt(math.pow(xc-txc,2) + math.pow(yc-tyc,2))
                
  
            # Extract patch, and resize
            self.class0patches[count,:,:] = resize(self.img[by:by+bh, bx:bx+bw], (self.CNN_SIZE, self.CNN_SIZE)) 

            # Draw patch rectangle
            if(self.SHOW_UPDATE_PATCHES):
                cv2.rectangle(self.visual, (bx,by),(bx+bw, by+bh), color=(0,0,200), thickness=1)        
                cv2.circle(self.visual, (xc, yc), 2, color=(0,0,255),thickness=1)

    def _grabClass1Patches(self):
        """
        Selects the training set of Class1 patches using Poisson dist        

        Returns:
            Loads slected patches into self.class1patches
        """

        # Choose indexes within class1 pool
        # Note: precomputed PMF saved in self.poissonPMF
        queueSize = self.class1pool.shape[0]
        patchInds = np.random.choice(range(0,queueSize), self.desNum0Patches, p=self.poissonPMF[queueSize-1,0:queueSize])

        for i in range(0,self.desNum0Patches):
            # Choose from clas1 pool, and save as patch
            self.class1patches[i,:,:,:] = self.class1pool[patchInds[i],:,:,:].copy()

    def _generatePoisson(self, length):
        """
        Fills array with Poisson PMF

        Returns a NxN matrix, where each row M is the PMF for M items
        (ie, when the queue_size is 30, use a scale PMF for 30 items: P_k[30, 0:30]) 
        Used in class-1 patch selection in Update Imager
        
        Args: 
            length: Maximum index for Poisson
        Returns:
            P_k:    NxN matrix, with rows as PMF for given index

        """

        lam = 1.0 # Poisson parameter (specified in paper)
        # Initialize PMF matrix
        P_k = np.zeros((length,length), float)
        # Loop through each row
        for row in range(0,length):
            # Loop through each index
            for index in range(0,row+1):
                k = int(math.floor((index/3)+1))
                # k = index+1
                # Poisson Equation
                P_k[row,index] = math.exp(-lam)*math.pow(lam,k) / math.factorial(k)
            # Normalize to unity sum (required for np.random.choice())
            P_k[row,:] = 1.0/np.sum(P_k[row,:]) * P_k[row,:]
        return P_k



