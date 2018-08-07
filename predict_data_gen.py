"""
predict_data_gen.py

    Generate prediction data for classification.
    Given an image, return possible patches, looking for target

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


class PredictDataGen():
    """
    Class implements patch generation for classification/prediction using trained CNN

    Given an input image, and previous known target location, this class extracts
    possible patches to search for the target. The possible patches are normally 
    distrubted in position and dimension based on last target bounding box.

    All possible patches are filtered using depth thresholding (see paper Fig 3)
    to minimize classification calls. Classification is not done here. This class
    merely 'returns' the valid patches that likely contain the target.
    """

    def __init__ (self, dims, SHOW_PATCH_SEARCH=False):
        """
        Initialization of Predict Imaging class
        """

        self.SHOW_PATCH_SEARCH = SHOW_PATCH_SEARCH

        self.IMG_W, self.IMG_H, self.IMG_C, self.INIT_W, self.INIT_H , self.CNN_SIZE = dims

        self.lastW = self.INIT_W
        self.lastH = self.INIT_H
        # Default params, should be overwritten before used
        self.lastX = self.IMG_W//2 - self.lastW//2
        self.lastY = self.IMG_H//2 - self.lastH//2
        self.lastD = 55


        # Depth Threshold values
        self.ALPHA = 0.25 * 255.0/6  # 0.25 m depth threshold
        self.THRESHOLD_CUTOFF = 0.7 # 70% of patch must be within ALPHA


    def generateData(self, img, LOST):
        """
        Does the work, generates the patches for classification

        Uses helper functions to narrow patch selection.
        
        Args:
            img:        nparray of RGBSD image
            LOST:       flag indicating if target is lost or not
        Returns:
            self.patches:  nparray of possible patches to pass through classifier (dim = M x 28x28 x 4)
            self.patchDims: nparray of dimensions of the corresponding patch      (dim = M x 4)
        """

        # self.ID = ID
        self.img = img
        self.LOST = LOST

        self.dIm = self.img[:,:,-1].copy() # Extract Depth channel of img

        if(self.SHOW_PATCH_SEARCH):
            self.visual = self.img.copy() # Create copy for drawing on

        # Generate binary threshold image
        if(not self.LOST): 
            self.depthBinary = self._depthThreshold()       

        # Array of possible patches, initialize with last patch dimensions
        self.patches = np.array([resize(self.img[self.lastY:self.lastY+self.lastH, self.lastX:self.lastX+self.lastW], (self.CNN_SIZE,self.CNN_SIZE))])
        self.patchDims = np.array([[self.lastX, self.lastY, self.lastW, self.lastH]])

        if(self.SHOW_PATCH_SEARCH):
            self.visual = self.img.copy() # Copy, to shade and show class-1 vs class-0

        # Search for valid patches
        if(not self.LOST):
            self._possiblePatches()
        else: # Lost, so search whole image
            self._lostPatches()

        # Return data as numpy
        return self.patches, self.patchDims

    def setLastPos(self, coords, d):
        """ 
        Update class params 

        Used externally in HumanTracker_CNN class
        """
        self.lastX, self.lastY, self.lastW, self.lastH = coords
        self.lastD = d

    def _depthThreshold(self):
        """
        Helper function returns binarized image using depth threshold
        """
        # Threshold image based on depth, 
        depBin = cv2.inRange(self.dIm, self.lastD-self.ALPHA, self.lastD+self.ALPHA, 255)
        if(self.SHOW_PATCH_SEARCH):
            cv2.imshow('depth binary', depBin)
        return depBin

    def _possiblePatches(self):
        """
        Generate possible patches localized to last target location and depth thresholded

        Randomly choses patches with similar coordinates and dimensions as last target 
        bounding box, then filters viable patches with depth thresholding.

        Depth thresholding explained in paper Fig 2. using >70% 
        """

        # std deviations for normal distributions
        std_x = 30 * self.IMG_W/672.0  # Adjust for varied spread
        std_y = 20 * self.IMG_H/376.0
        std_w = 15 * self.IMG_W/672.0
        std_h = 15 * self.IMG_H/376.0

        minWidth = 60 * self.IMG_W/672.0
        minHeight = 80 * self.IMG_H/376.0

        # Aim for desNumPatches, but don't run forever (only check totalTries)
        desNumPatches = 10 # How many patches to send to classified
        countGoodPatches = 0 # Count how many pass depth threshold
        totalTries = 150
        numTries = 0

        
        # Loop through patches
        while (countGoodPatches < desNumPatches and numTries < totalTries):
            numTries += 1

            # Generate distributed patch dimensions
            bx = int(np.random.normal(self.lastX, std_x))
            by = int(np.random.normal(self.lastY, std_y))
            bw = int(np.random.normal(self.lastW, std_w))
            bh = int(np.random.normal(self.lastH, std_h))

            # Edge cases
            if(bx < 0): # Left edge
                bx = 0
            if(by < 0): # Top edge
                by = 0

            # Don't let it get too skinny
            if(bw < minWidth):
                bw = minWidth
            if(bh < minHeight):
                bh = minHeight

            if(bx+bw >= self.IMG_W): # Right edge 
                bw = self.IMG_W-bx-1
            if(by+bh >= self.IMG_H): # Bottom edge
                bh = self.IMG_H-by-1

            bw, bh = int(bw), int(bh)

            # Check if over 70% of patch is in depth range
            if(np.mean(self.depthBinary[by:by+bh, bx:bx+bw])/255.0 > self.THRESHOLD_CUTOFF):
                # Extract and resize patch
                newpatch = resize( self.img[by:by+bh , bx:bx+bw], (self.CNN_SIZE,self.CNN_SIZE))
                # Append to arrays
                self.patches = np.append(self.patches, np.array([newpatch]), axis=0)
                self.patchDims = np.append(self.patchDims, np.array([[bx, by, bw, bh]]), axis=0)
                countGoodPatches += 1 # increment count
                # Draw green patch
                if(self.SHOW_PATCH_SEARCH):
                    cv2.rectangle(self.visual, (bx,by),(bx+bw, by+bh), color=(0,100,0), thickness=1)
                    cv2.circle(self.visual, (bx+bw//2,by+bh//2),2,(0,255,0),thickness=1)
                    cv2.imshow('visual',self.visual)
                    cv2.waitKey(10)                
            else:
                # Draw red patch
                if(self.SHOW_PATCH_SEARCH):
                    cv2.rectangle(self.visual, (bx,by),(bx+bw, by+bh), color=(0,0,100), thickness=1)
                    cv2.circle(self.visual, (bx+bw//2,by+bh//2),2,(0,0,255),thickness=1)
                    cv2.imshow('visual',self.visual)
                    cv2.waitKey(10)    
        # Show all patches
        if(self.SHOW_PATCH_SEARCH):
            cv2.imshow('visual',self.visual)
            cv2.waitKey(1000)


    def _lostPatches(self):
        """
        Grid search the whole image for target using fixed box size.

        """

        horzSteps = 15
        vertSteps = 5
        # horzStepSize = (self.IMG_W - self.INIT_W)//horzSteps
        # vertStepSize = (self.IMG_H - self.INIT_H)//vertSteps
        horzStepSize = (self.IMG_W - self.lastW)//horzSteps
        vertStepSize = (self.IMG_H - self.lastH)//vertSteps
        

        bw, bh = self.lastW, self.lastH
        # bw, bh = self.INIT_W, self.INIT_H

        # Grid searc
        for row in range(vertSteps):
            by = row*vertStepSize
            for col in range(horzSteps):
                bx = col*horzStepSize

                # Extract and resize patch
                newpatch = resize(self.img[by:by+bh, bx:bx+bw], (self.CNN_SIZE, self.CNN_SIZE))
                # Append to arrays
                self.patches = np.append(self.patches, np.array([newpatch]), axis=0)
                self.patchDims = np.append(self.patchDims, np.array([[bx, by, bw, bh]]), axis=0)

                # Draw patches
                if(self.SHOW_PATCH_SEARCH):
                    cv2.rectangle(self.visual, (bx,by),(bx+bw, by+bh), color=(0,100,0), thickness=1)
                    cv2.circle(self.visual, (bx+bw//2,by+bh//2),2,(0,255,0),thickness=1)

        # Show all patches
        if(self.SHOW_PATCH_SEARCH):
            cv2.imshow('visual',self.visual)
            cv2.waitKey(1000)

