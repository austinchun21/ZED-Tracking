# human-tracking-zedd
Contract Project: Human Tracking with Zedd Camera

Client: Leo

Recreate "Integrating Stereo Vision with a CNN Tracker for a Person-Following Robot". 
https://www.raghavendersahdev.com/person-following-robot-cnn.html


Hardware Requirements
- Jetson TX2
- Zed Camera

Software Requirements
- JetPack 3.2 L4T-28.2.1 (CUDA 9.0, cuDNN 7.0.5)
- TensorFlow
- Keras
- ROS
- Zed ROS Library


Jetson Software Versions
- TF: 1.9.0-rc0
- Keras: 2.2.0
- Python: 2.7.12

Seting up Jetson Software
- Flash Jetson TX2 with JetPack3.2
	- Includes os, root stuff needed for ML applications
	- Need an Ubuntu 14.04 (16.04 can work) host computer to flash
	- NVIDIA Guide: https://docs.nvidia.com/jetpack-l4t/index.html#jetpack/3.2.1/install.htm
	- Jetson Hacks: https://www.jetsonhacks.com/2017/03/21/jetpack-3-0-nvidia-jetson-tx2-development-kit/
	- Test CUDA install (and look at pretty graphcis)
		- `cd ~/NVIDIA_CUDA-9.0_SAMPLES/bin/aarch64/linux/release`
		- `./oceanFFT` or `./particles`
- Install TensorFlow
	- Make sure to do a `sudo apt-get update && sudo apt-get dist-upgrade`
	- Use master branch so just `sudo bash BuildTensorflow.sh` in ~/JetsonTFBuild
	- Jetson Hacks: https://www.jetsonhacks.com/2018/03/26/build-tensorflow-on-nvidia-jetson-tx-development-kits/
	- Test with <Hello world>
- Install Keras
	- Can't just do `sudo pip install keras`
		- pip doesn't seem to work to well for the requirements
		- Instead need to manually `apt-get`
	- `sudo -H pip install --upgrade pip`	(Gets from 8.1.1 to 10.0.1)
	- `sudo apt autoremove` (Just clean up a little)
	- `sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose`
	- `sudo pip install pyyaml`
	- `sudo apt-get install python-h5py`
- Test Keras
	- Run cifar10 image Classification CNN
	- `git clone https://github.com/keras-team/keras.git`
	- `cd keras/examples`
	- `python cifar10_cnn.py`
- Install ROS
- Install ZED ROS wrapper
    - sudo apt-get install ros-kinetic-image-view       (image_view didn't work for me)
    - sudo apt-get install python-cv-bridge     (for converting ROS msg Image to cv numpy array)


Running ZED ROS Wrapepr
- Need multiple terminals
- First terminal
    - `roscore`
- Next terminal
    - `cd jetsonbot`    (or whatever catkin_ws)
    - `source devel/setup.bash`
    - `roslaunch zed_wrapper zed.launch`
        - Launches the ZED wrapper without the gui, just publishes to topics
        - 
    


CNN Overview

- Initialization
    - From initial box assume center is truth data (class-1)
    - Randomly select 40 class-0 patches
    - Either
        - Copy single class-1 patch 40 times
        - or Train with single class-1 patch with a weight of 40

- Testing
    - Take in new image
    - Using last target location, do local search (normal distribution)
    - Depth threshold
    - Classify remaining patches
    - Return patch with MAX prediction

- Update
    - When search (testing phase) locates the target with high confidence
    - Add the target patch to a pool of class-1 patches
    - Sample class-1 patches from pool of last 50 patches based on 
    - Extract 40 random patches of surroundings patches as class-0
    - 
