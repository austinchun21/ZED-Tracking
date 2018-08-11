"""
ZED_Tracking_Algorithm.py

    Human tracking algorithm interfacing with ZED depth camera. Uses YOLOv2 (tiny) for
    human detection (and localization), along with a lightweight CNN that is trained
    on initialization to distinguish between the desired target, and other people.

    Instructions to Run:
        - Run ZED ROS wrapper
            - In a terminal
                > roscore
            - In another terminal
                > cd ~/jetsonbot
                > source devel/setup.bash
                > roslaunch zed_wrapper zed.launch
        - Run ZED Tracker Algorithm
            - In a new terminal, run python script
                > python ZED_Tracking_Algorithm.py

    Options:
        - To skip initialization steps:
            > python ZED_Tracking_Algorithm.py --skipInit

    The algorithm uses a (simple linear/one-way) state-transition system for operation
    Algorithm States:
        Waiting 1
            - Shows live feed of ZED RGB feed
            - Instruct user to how to start (press 'i')
        Countdown 1
            - Countdown 5 seconds, allow user to position themself in front of camera
            - Show initializationa cceptable area
        Init 1
            - Capture N images of Target (for training CNN)
            - Assumes Target is in the middle, and the only one in the middle
        Waiting 2
            - User should now exit camera view
            - Press 'b' to continue
        Countdown 2
            - Wait a couple seconds before starting 'Not Target' sample selection
        Init 2
            - Gather any patches of 'other' humans to store as 'Not Target' for training
            - 'Other' humans are chose across whole frame (not restricted to middle like Init 1)
            - Assumes target is not anywhere in frame
            - Only captures fo say 3 seconds
        Init 3
            - Capture photo of environment
            - Use grided pattern for patches as 'Not Target'
                - Gives a sense of environment that also isn't the target
        Train
            - Blocking function call that trains CNN on training data (saved in train/ directory)
            - Shows a plot of training progress (loss, and accuracy)
        Predict
            - This is where the Tracker actually does the tracking
            - Uses YOLO to detect all humans
            - Passes all 'person' objects to CNN to classify
            - Designates highest classication as target

    Training Data:
        - Training data saved to directory during Init 1-3
        - Training data read from directory duing CNN.train()
        - Training data consists of: (assuming NUM_INIT_DATA = 100)
            - 100 Target samples, taken during Init 1
            - About 100+ Non-Target samples
                - 50 samples from generic Pedestrian database
                - Variable amount of samples of 'others' taken during Init 2
                - 45 sampels from static background shot
        - Training data example configuration
            - ZED-Tracking/
                - ZED_Tracking_Algorithm.py
                - CNN.py
                - darkflow/
                - data/ 
                    - train/
                        - target/
                            - Target-####.png
                            - Target-####.png
                            - Target-####.png
                                ...
                        - not_target/
                            - ###.png       # images from Pedestrian dataset
                            - ###.png
                            - ###.png
                                ...
                            - Other-####.png    # images of other people
                            - Other-####.png
                            - Back-####.png     # images of background
                                ...

    Typical Hyperparameters:
        - Initial Learning Rate = 1e-3
        - Epochs = 20
        - Batch Size = 50

    Notes:
        - Tracker publishes depth, angle, and targetLost data to ROStopics
            - targetLost topic is continuously posting
            - depth and angle only post when valid values (can change in code)

    Author: Austin Chun  <austinchun21@gmail.com>
    Date:   August 11th, 2018


"""


# General libraries
import cv2
import numpy as np
import math
from time import time
import matplotlib.pyplot as plt
import argparse
import sys
# ROS
import rospy
from sensor_msgs.msg import Image # , Float32
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge, CvBridgeError
# Darkflow
from darkflow.net.build import TFNet
# Import Custom CNN
from CNN import TargetCNN

###########################
### User Config Options ###
###########################
time_till_init = 5.0            # Countdown between pressing 'i' and initialization start    
time_till_init_2 = 3.0          # Countdown between pressing 'b' and background capture 
time_till_init_3 = 3.0          # Duration to capture other people
pred_thresh = 0.5               # Threshold below which the target is considered missing (between 0 and 1)

trainDir = 'data/train/'                # Training data directory
targetDir = trainDir+'target/'          # Subdirectory in training, holds target images
notTargetDir = trainDir+'not_target/'   # Subdirectory in training, holds noy_target images

NUM_INIT_DATA = 100 # Number of images to capture of Target, for CNN Training 
INIT_WIN_W = 150    # Width of the initialization 'acceptable' area (pixels)

########################
### Global Variables ###
########################
FONT = cv2.FONT_HERSHEY_SIMPLEX # font for displaying text
IMG_W = 1280        # img width, pixels
IMG_H = 720         # img height, pixels
FOV = 90.0          # 90 deg Field of View for ZED camera
MID_X, MID_Y = IMG_W//2 , IMG_H//2 # Middle (x,y)

bridge = CvBridge() # used for converting ROS msg images, to nparrayts

depthImg = np.zeros((IMG_H,IMG_W,1))    # Store depth image from ZED callback
bgrImg   = np.zeros((IMG_H,IMG_W,3))    # Store BGR image from ZED callback

########################
###  Main Algorithm  ###
########################
def main():
    """
    Interface with a ZED depth camera (via ROS), using YOLO for object detection, 
    and a custom CNN for target classification.
    """

    # Argument parser for command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--skipInit",
        help="Flag whether to skip initialization or not",
        action='store_true')
    args = vars(ap.parse_args())


    # TF Options (either build from config/weights, or load pre-built from pb/meta)
    # # Build tiny-yolo from the config and weights
    # options = {"model": "darkflow/cfg/tiny-yolo-voc.cfg", 
    #            "load":"darkflow/bin/tiny-yolo-voc.weights", 
    #            "threshold":0.5, "gpu":0.5}
    # Load the pre-trained and pre-built model of tiny-yolo
    options = {"pbLoad": "darkflow/built_graph/tiny-yolo-voc.pb", 
             "metaLoad": "darkflow/built_graph/tiny-yolo-voc.meta",
             "threshold":0.5, "gpu":0.4}
    
    # Initialize YOLO 
    tfnet = TFNet(options)
    # Initialize CNN
    CNN = TargetCNN()
 

    # ROS Setup
    rospy.init_node('ZED_test', anonymous=True)
    # Subscribe to topics
    depth_sub = rospy.Subscriber("/zed/depth/depth_registered", Image, depth_cb, queue_size=1)
    left_sub  = rospy.Subscriber("/zed/left/image_rect_color",  Image, bgr_cb, queue_size=1)
    # Publish depth, angle, and target lost
    depth_pub = rospy.Publisher("/ZED_Tracker/depth", Float32, queue_size = 10)
    angle_pub = rospy.Publisher("/ZED_Tracker/angle", Float32, queue_size = 10)
    lost_pub = rospy.Publisher("/ZED_Tracker/targetLost", Bool, queue_size = 10)

    # Thickness of cv draw objects
    thick = int((IMG_H + IMG_W) // 300)

    # Initialize variables
    lastTime = time()
    frame_count = -1        # index video frame
    init_target_count = 0   # number of Target samples captured
    other_count = 0         # number of Others(other people, not the Target) sample captured
    xc,yc = 0,0             # Centroid of Bounding Box

    # Keep track of what 'state' the algorithm is in
    state = 'WAITING_1'     # First state, wait for user 
    # Check if skipInit
    if(args["skipInit"]):
        state = "TRAIN"

    print("  State: %s"%state)

    #############################
    ###  Main Processing Loop ###
    #############################
    while(not rospy.is_shutdown()):
        frame_count += 1

        if(state == 'TRAIN'): # Training is a blocking function call, so visuals don't update
            print("[INFO] Starting training")
            CNN.train(PLOT_LEARNING=True, epochs=20, trainDir=trainDir)
            print("[INFO] Done Training")
            state = 'PREDICT'
            print("  State: %s"%state)

        # Use YOLO for Object Detection
        results = tfnet.return_predict(bgrImg) # predict using bgrImg from ROS callback

        vis = bgrImg.copy() # Create copy for visual

        # Initialize max values 
        max_pred = 0                    # Store max prediction value
        max_pred_coords = [0,0,0,0]     # Store corresponding coords

        # Loop through all detections (from YOLO)
        for i in range(len(results)):
            # Skip any non-'person' detections
            if(results[i]['label'] != 'person'):
                continue
            # Bounding Box indexes
            x1, y1 = results[i]['topleft']['x'], results[i]['topleft']['y']
            x2, y2 = results[i]['bottomright']['x'], results[i]['bottomright']['y']
            xc, yc = toCentroid(x1,y1,x2,y2) # Get centroid
            
            # Draw Bounding Box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255,255,255), thick)
            cv2.circle   (vis, (xc,yc), 3, (255,255,255), thick)    # Center circle
            cv2.putText(vis, results[i]['label'], (x1 + thick*2, y1 + thick*2), 0, 1e-3 * IMG_H, (255,255,255),thick//3)

            if(state == 'INIT_1'):
                # Only consider objects within middle window 
                # (Center window extends full top to bottom, but only +/- INIT_WIN_W)
                if(abs(xc - MID_X) < INIT_WIN_W):
                    obj = bgrImg[y1:y2, x1:x2].copy()
                    obj_name = targetDir + "Target-%s.png" %(str(init_target_count).zfill(4))
                    init_target_count += 1
                    cv2.imwrite(obj_name, obj)      

            if(state == 'INIT_2'):
                # Save any person, no restriction on location
                other = bgrImg[y1:y2, x1:x2].copy()
                other_name = notTargetDir + "Other-%s.png" %(str(other_count).zfill(4))
                other_count += 1
                cv2.imwrite(other_name, other)
            
            if(state == 'PREDICT'):
                # Use trained CNN to classify as Target/Not-Target
                label, prob, prediction  = CNN.predict(bgrImg[y1:y2, x1:x2].copy())
                # Keep track of highest classification
                if(prediction > max_pred):
                    max_pred = prediction
                    max_pred_coords = [x1,y1,x2,y2]

        # Wait 1ms (allows visuals to update), and take in user input if any
        charin = cv2.waitKey(1)

        ## Draw State specific graphics/instructions/info
        ## and transition states when appropriate

        if(state == 'WAITING_1'):
            # Draw initialization box area
            cv2.rectangle(vis, (MID_X-INIT_WIN_W, 0), (MID_X+INIT_WIN_W, IMG_H), (0,150,0), thick//3)
            # Show instructions on screen
            instr = ["Press 'i' to continue,","then you will have %d seconds"%time_till_init,"before initialization starts"]
            cv2.rectangle(vis, (0,0),(300,100), color=(255,255,255),thickness=-1) # box for text
            cv2.putText(vis,instr[0],(20,20), FONT, 0.5 ,(0,0,255),2) # Show instructions
            cv2.putText(vis,instr[1],(20,40), FONT, 0.5 ,(0,0,255),2) # Show instructions
            cv2.putText(vis,instr[2],(20,60), FONT, 0.5 ,(0,0,255),2) # Show instructions
            # User press 'i' to move to initialization
            if(charin == ord('i')):
                cdown_start = time()    # Start the time
                state = 'COUNTDOWN_1'
                print("  State: %s"%state)
        if(state == 'COUNTDOWN_1'):
            # Draw initialization box area
            cv2.rectangle(vis, (MID_X-INIT_WIN_W, 0), (MID_X+INIT_WIN_W, IMG_H), (0,150,0), thick//3)
            # Countdown time, wait for X seconds before capturing init data
            dur = time() - cdown_start
            cv2.putText(vis,"%d" %(time_till_init-(dur)+1),(IMG_W//4,IMG_H//4), FONT, 4 ,(0,0,255),2) # Show countdown
            if(dur > time_till_init):
                state = 'INIT_1'
                print("  State: %s"%state)
                print("     Saving Target samples as: %sTarget-####.png"%targetDir)
        if(state == 'INIT_1'):
            # Draw initialization box area
            cv2.rectangle(vis, (MID_X-INIT_WIN_W, 0), (MID_X+INIT_WIN_W, IMG_H), (0,150,0), thick//3)
            # Display progress on image acquisition
            cv2.rectangle(vis, (0,0),(400,45), color=(255,255,255),thickness=-1) # box for text
            cv2.putText(vis,"%d of %d images taken"%(init_target_count, NUM_INIT_DATA),(10,30), FONT, 1 ,(0,0,255),2) # Show instructions
            # Continue once X samples are taken
            if(init_target_count >= NUM_INIT_DATA):
                state = 'WAITING_2'
                print("  State: %s"%state)
        if(state == 'WAITING_2'):
            # Show instructions on screen
            instr = ["Press 'b' to continue,","then you will have %d seconds"%time_till_init_2,"before the background capture starts"]
            cv2.rectangle(vis, (0,0),(300,100), color=(255,255,255),thickness=-1) # box for text
            cv2.putText(vis,instr[0],(20,20), FONT, 0.5 ,(0,0,255),2) # Show instructions
            cv2.putText(vis,instr[1],(20,40), FONT, 0.5 ,(0,0,255),2) # Show instructions
            cv2.putText(vis,instr[2],(20,60), FONT, 0.5 ,(0,0,255),2) # Show instructions
            # User press 'b' to move to background initialization
            if(charin == ord('b')):
                cdown_start = time()
                state = 'COUNTDOWN_2'
                print("  State: %s"%state)
        if(state == 'COUNTDOWN_2'):
            dur = time() - cdown_start
            cv2.putText(vis,"%d" %(time_till_init_2-(dur)+1),(IMG_W//4,IMG_H//4), FONT, 4 ,(0,0,255),2) # Show countdown
            # Wait for X seconds before capturing init data
            if(dur > time_till_init_2):
                cdown_start = time()
                state = 'INIT_2'
                print("  State: %s"%state)
                print("     Saving Others samples as: %sOther-####.png"%notTargetDir)
        if(state == 'INIT_2'):
            dur = time() - cdown_start
            # Wait for X seconds before capturing background
            if(dur > time_till_init_3):
                state = 'INIT_3'
                print("  State: %s"%state)
        if(state == 'INIT_3'):
            # Save 'background' patches to notTarget training directory
            grabBackground(bgrImg) 
            msg = "Please wait while the CNN is trained..."
            cv2.rectangle(vis, (0,0),(IMG_W,100), color=(255,255,255),thickness=-1) # box for text
            cv2.putText(vis,msg,(50,50), FONT, 2 ,(0,0,255),2) # Show instructions
            cv2.imshow('ZED Tracking', vis)
            cv2.waitKey(1)
            state = 'TRAIN'
            print("  State: %s"%state)
        if(state == 'PREDICT'):
            # Display max classification
            x1,y1,x2,y2 = max_pred_coords
            cv2.rectangle(vis, (x1,y1),(x2,y2), color=(0,255,0), thickness=thick)
            cv2.rectangle(vis, (0,0),(300,160), color=(255,255,255), thickness=-1)
            cv2.putText(vis, "Pred: %.3f"%max_pred, (10,30), FONT, 1, color=(255,0,0), thickness=2)
            xc,yc = toCentroid(x1,y1,x2,y2)

            # Display and publish Depth
            depth = getDepth(xc,yc, depthImg)
            if(not np.isnan(depth)):
                cv2.putText(vis, "Depth: %.2f m"%depth, (10,60), FONT, 1, color=(255,0,0), thickness=2)
                # Publish depth
                depth_pub.publish(depth)
            else:
                cv2.putText(vis, "Depth: ---- m", (10,60), FONT, 1, color=(255,0,0), thickness=2)
                ## Uncomment line below if you want constant publishing of depth data 
                # depth_pub.publish(-1.0) # -1.0 is simply an 'error' value
            
            # Display and publish Angle            
            angle = getAngle(xc)
            if(not angle == -45.0):
                cv2.putText(vis, "Angle: %.1f deg"%angle, (10,90), FONT, 1, color=(255,0,0), thickness=2)
                angle_pub.publish(angle)
            else:
                cv2.putText(vis, "Angle: ---- deg", (10,90), FONT, 1, color=(255,0,0), thickness=2)
                ## Uncomment line below if you want constant publishing of angle data 
                # angle_pub.publish(-180.0) # -180.0 is simply an 'error' value

            # Target Lost
            LOST = max_pred < pred_thresh
            if(LOST):
                cv2.putText(vis, "TARGET LOST", (10,150), FONT, 1.4, color=(0,0,255), thickness=2)
            lost_pub.publish(LOST)


        # Press 'q' to quit
        if(charin == ord('q')):
            print("Quitting program...")
            break

        # FPS calculation
        f = 1/(time() - lastTime)
        fps = "FPS: %.2f" %f
        lastTime = time()
        # Display FPS data
        cv2.rectangle(vis,  (IMG_W-170,IMG_H-50), (IMG_W,IMG_H), (0,0,0), -1)
        cv2.putText(vis,fps, (IMG_W-160, IMG_H-15), FONT, 1, (0,0,255), 2)

        # Show visuals
        cv2.imshow('ZED Tracking', vis)

    # Cleanup
    cv2.destroyAllWindows()
    depth_sub.unregister()
    left_sub.unregister()


########################
### Helper Functions ###
########################

def depth_cb(data):
    """
    ROS subscriber callback function
    Read depth data from ros message
    """
    try:
        global depthImg
        # Convert Image msg to cv2 (nparray)
        depthImg = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough") 
        # Convert float32 to uint8 (scale from 0-6 meters, to 0-255 greyscale)
        depthImg = (255./6 * depthImg).astype(np.uint8)
    except CvBridgeError as e:
        print(e)

def bgr_cb(data):
    """
    ROS subscriber callback function
    Read BGR image data from ros message
    """
    try:
        global bgrImg
        # Convert Image msg to cv2 (nparray)
        bgrImg = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

def toCentroid(x1,y1,x2,y2):
    """ Return centroid from bounding box coordinates"""
    return (x1+x2)//2, (y1+y2)//2

def grabBackground(img):
    """
    Given a frame, save patches of the background to the 'not_target/' folder for training.
    Inputs:
        img:            frame from video, RGB, nparray, full video size
    Returns:
        --- Just saves patches to directory
    """
    # Fixed patch dimensions    (can play around with)
    width = 150
    height = 300
    x_steps = 9 # num horizontal steps
    y_steps = 5 # num vertical steps
    dx = (IMG_W-width)//x_steps
    dy = (IMG_H-height)//y_steps

    print("     Saving background samples as: %sBack-####.png"%notTargetDir)

    back_count = 0 # Count number of background samples
    # Grab patches in grid pattern
    for row in range(y_steps):
        for col in range(x_steps):
            x,y = col*dx, row*dy
            back = img[y:y+height, x:x+width].copy()
            back_name = notTargetDir + "Back-%s.png"%(str(back_count).zfill(4))
            back_count += 1
            cv2.imwrite(back_name, back)

def getDepth(xc,yc,dImg):
    """ 
    Extract depth data from ZED depth image. Takes the average of a local area
    Inputs:
        xc, yc:         centroid coordinates of Target bounding box
        dImg:           depth image from ZED rostopic
    Returns:
        depth:          depth to Target in meters
    """
    r = 20  # Radius of local average
    local = dImg[ (yc-r):(yc+r) , (xc-r):(xc+r)].copy()
    depth = np.mean(local[np.isfinite(local)]) # Only take mean of finite values (avoid NAN)
    depth = depth * 6.0/255  # Convert grayscale to 
    return depth

def getAngle(xc):
    """
    Calculate angle (horizontally) to target (from middle of screen)
    """
    angle = (xc - MID_X) * FOV / IMG_W # degrees (b/c FOV in degrees)
    return angle




if __name__ == '__main__':
    main()
