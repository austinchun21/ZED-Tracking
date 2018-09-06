# ZED-Tracking
Contract Project: ZED Human Tracking on Jetson TX2

Client: Leo

Human tracking algorithm interfacing with ZED depth camera. Uses YOLOv2 (tiny) for
human detection (and localization), along with a lightweight CNN that is trained
on initialization to distinguish between the desired target, and other people.
Algorithm publishes the Target's depth and angle to ROS topics.

### Hardware Requirements
- Jetson TX2
- ZED Depth Camera

### Software Requirements
- JetPack 3.2 L4T-28.2.1 (CUDA 9.0, cuDNN 7.0.5)
- TensorFlow
- Keras
- ROS
- Zed ROS Library

### Jetson Software Versions
- TF: 1.9.0-rc0
- Keras: 2.2.0
- Python: 2.7.12

## Instructions to Run:
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

### Options:
- To skip initialization steps:
    > python ZED_Tracking_Algorithm.py --skipInit

## Algorithm States:
The algorithm uses a (simple linear/one-way) state-transition system for operation
- Waiting 1
    - Shows live feed of ZED RGB feed
    - Instruct user to how to start (press 'i')
- Countdown 1
    - Countdown 5 seconds, allow user to position themself in front of camera
    - Show initializationa cceptable area
- Init 1
    - Capture N images of Target (for training CNN)
    - Assumes Target is in the middle, and the only one in the middle
- Waiting 2
    - User should now exit camera view
    - Press 'b' to continue
- Countdown 2
    - Wait a couple seconds before starting 'Not Target' sample selection
- Init 2
    - Gather any patches of 'other' humans to store as 'Not Target' for training
    - 'Other' humans are chose across whole frame (not restricted to middle like Init 1)
    - Assumes target is not anywhere in frame
    - Only captures fo say 3 seconds
- Init 3
    - Capture photo of environment
    - Use grided pattern for patches as 'Not Target' (Gives a sense of environment that also isn't the target)
- Train
    - Blocking function call that trains CNN on training data (saved in train/ directory)
    - Shows a plot of training progress (loss, and accuracy)
- Predict
    - This is where the Tracker actually does the tracking
    - Uses YOLO to detect all humans
    - Passes all 'person' objects to CNN to classify
    - Designates highest classication as target

## Training Data:
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

## Typical Hyperparameters:
- Initial Learning Rate = 1e-3
- Epochs = 20
- Batch Size = 50

## Notes:
- Tracker publishes depth, angle, and targetLost data to ROStopics
    - targetLost topic is continuously posting
    - depth and angle only post when valid values (can change in code)
- For all times when user needs to press a key, the visual must be selected
  meaning the user must be clicked on the window with the video feed to 
  have the key be registered
- Press 'q' to quit the program from the visual
