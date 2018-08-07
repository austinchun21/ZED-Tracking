"""
initial_data_gen.py

    Generate initial data for training CNN

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


class InitialDataGen():
    """
    @brief      Class for initial image parser.

    This class generates the Initial test data set from an input image.
    The test data are patches selected from regions of a single image.
    A single Class-1 patch is assumed to be at the center, 100x350 pixels
    Class-0 patches are uniformly randomly selected such that they do not
    overlap with the Class-1 patch in the middle. All patches are resized
    to 28x28 for CNN input.

    The class ultimately returns an array of class-1 and class-0 patches (28x28)
    along with corresponding labels.

    Following the paper, this uses N=40 class-0 patches, and N=40 class-1 patches.

    The selection of class-1 patches is being experimented with. Either a single 
    patch, copied 40 times (as the paper suggests), or possibly multiple patches 
    from the first couple frames, or using a single/few class-1 patches (no copying)
    along with weighting in the Keras training function (class_weights).
    """

    def __init__ (self, img, dims, SHOW_INIT_PATCHES=False):
        """
        Initialization requires Image ID number (assumes using dataset from paper).
        Assumes target is located in the initial bounding box of 100x350.
    
        Args:
            img:    RGBSD numpy array   [self.IMG_W x self.IMG_H x self.IMG_C]
        Kwargs:
            ID:     img ID number, to index using dataset
            SHOW_INIT_PATCHES: Displays the chosen class-1 and class-0 patches selected
        """
 
        # Store image 
        self.img = img

        self.IMG_W, self.IMG_H, self.IMG_C, self.box_w, self.box_h , self.CNN_SIZE = dims

        self.SHOW_INIT_PATCHES = SHOW_INIT_PATCHES


        # Assume target is centered, in bounding box
        self.x1 = self.IMG_W//2 - self.box_w//2
        self.x2 = self.IMG_W//2 + self.box_w//2
        self.y1 = self.IMG_H//2 - self.box_h//2
        self.y2 = self.IMG_H//2 + self.box_h//2

        # N = 40, as per the paper
        self.desNum0Patches = 40      

        # Default size: 40 (patches) x 28x28(pixels) x 4(channels, RGBD) 
        self.class0patches = np.empty((self.desNum0Patches,self.CNN_SIZE,self.CNN_SIZE,self.IMG_C))
        self.class1patches = np.empty((0,self.CNN_SIZE,self.CNN_SIZE,self.IMG_C))

        # Generate the data on initialization (where all the work happens)
        self._generateInitData()

    def _generateInitData(self):
        """
        Does the work, calling helper functions to actually return the data
        
        Loads image based on ID. Selects Class-1 patches, then Class-0 patches.
        Appends patches and creates labels to return. 

        Saves patches and labels to class variable
            patches:    nparray of patches (class-1 then class-0 appended)
            labels:     nparray of corresponding labels (1 or 0)
        """

        if(self.SHOW_INIT_PATCHES):
            self.visual = self.img.copy() # Create copy for drawing on

        # Chose Patches
        self._grabClass1Patch()

        # Grab patches of background
        self._grabClass0Patches()

        # Show selected patches
        if(self.SHOW_INIT_PATCHES):
            cv2.imshow('visual',self.visual)
            cv2.waitKey(1000)    

        # Append patches/labels to one array
        self.patches = np.append(self.class1patches, self.class0patches, axis=0)    # ex (40+40 x 28x28 x 4)
        self.labels  = np.append(np.ones((self.class1patches.shape[0],1)),np.zeros((self.desNum0Patches,1)), axis=0) # ex (40+40 x 1)

    def _grabClass1Patch(self):
        """
        Extracts assumed class-1 patch(es)
        
        Initialization assumes target is in initial bounding box
        at the center of the image, with predetermined width/height.

        Adds the selected patch(es) to class variables.
        Optionally can configure to extract class-1 patches from first
        couple images (rather that just the first), or can add 
        multiples of the patches 
        """

        # Assumed centroid
        xc, yc = self.IMG_W//2, self.IMG_H//2
        # Draw Class-1 patch on visual
        if(self.SHOW_INIT_PATCHES):
            cv2.circle(self.visual, (xc,yc),25,color=(0,100,0),thickness=-1)
            cv2.circle(self.visual, (xc,yc),2,color=(0,255,0),thickness=1)


        meanWidth = self.box_w
        meanHeight = self.box_h
        stdW = 10 * self.IMG_W/672.0
        stdH = 10 * self.IMG_H/376.0

        # Choose 40 patches with the same centroid, but varying width/heights
        for _ in range(self.desNum0Patches):
            bw = int(np.random.normal(meanWidth, stdW))
            bh = int(np.random.normal(meanHeight, stdH))

            patch = resize(self.img[yc-bh//2:yc+bh//2 , xc-bw//2:xc+bw//2], (self.CNN_SIZE, self.CNN_SIZE) )
            self.class1patches = np.append(self.class1patches, [patch], axis=0)

            # Draw Class-1 patch on visual
            if(self.SHOW_INIT_PATCHES):
                cv2.rectangle(self.visual, (xc-bw//2,yc-bh//2),(xc+bw//2,yc+bh//2), color=(0,255,0), thickness=1)

    def _grabClass0Patches(self):
        # Use normal distribution, with empty space in the center of target
        
        # Center coordinates (assumed to be true centroid)
        txc, tyc = self.IMG_W//2, self.IMG_H//2
        stdX = 60 * self.IMG_W/672.0
        stdY = 20 * self.IMG_H/376.0

        # Set mean and std for varying bounding box sizes
        meanWidth, meanHeight = self.box_w, self.box_h
        stdWidth = 30 * self.IMG_W/672.0 # Adjustable std dev (pixels)
        stdHeight = 30 * self.IMG_H/376.0

        radiusThreshold = 30 * self.IMG_W/672.0

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
            if(self.SHOW_INIT_PATCHES):
                cv2.rectangle(self.visual, (bx,by),(bx+bw, by+bh), color=(0,0,200), thickness=1)        
                cv2.circle(self.visual, (xc, yc), 2, color=(0,0,255),thickness=1)
