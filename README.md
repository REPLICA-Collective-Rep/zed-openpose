# OpenPose ZED Extractor

This is a forked example project used for extracting pose and skeleton data into JSON, with an accompanying bash script for automation. It has been used on a Paperspace VM with over 120 hours of footage! 

## Build

Building executable in QTCreator rather than CMake:

1. Use `CMakeLists.txt` as project config in QT
2. Set environment variables inside the project config: **Projects > Build & Run > Run > Run Environment** , `LD_LIBRARY_PATH` to `$LD_LIBRARY_PATH:/path/to/cuda-10.0/lib64:/path/to/openpose/caffe/lib`
3. Add command line arguments for testing in **Projects > Build & Run > Run**, ie. `-net_resolution 320x240 -model_pose MPI_4_layers  -svo_path /path/to/my/file.svo`


## Usage

The script `automator.sh` will loop through a directory of SVOs, generating JSON files, and moving completed and failed extractions to `completed` and `failed` folders. Email updates will be sent via curlmail.co if an email address is supplied.

```

# -s [source directory of SVO files]
# -c [clear log.txt file when the script begins]
# -t [create and test dummy SVO files]
# -e [email address for sending log.txt notifications]
# -i [email frequency, ie. send email every nth file]

./automator.sh -s ~/Desktop/bin/03_ZED/ -c -e gilbert.sinnott@mailbox.org


```

# OpenPose ZED


<p align="center">
    <img src="OpenPose_ZED.gif", width="800">
</p>

This sample show how to simply use the ZED with [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), the deep learning framework that detects the skeleton from a single 2D image. The 3D information provided by the ZED is used to place the joints in space.
The output is a 3D view of the skeletons.

## Installation

### Openpose

This sample can be put in the folder [`examples/user_code/`](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/user_code) **OR** preferably, compile and _install_ openpose with the cmake and compile this anywhere

The installation process is very easy using cmake.

Clone the repository :

        git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose/
        
Build and install it :
        
        cd openpose
        mkdir build
        cmake .. # This can take a while
        make -j8
        sudo make install


### ZED SDK

The ZED SDK is also a requirement for this sample, [download the ZED SDK](https://www.stereolabs.com/developers/release/latest/) and follows the [instructions](https://docs.stereolabs.com/overview/getting-started/installation/).

It requires ZED SDK 2.4 for the floor plane detection but can be easily [disabled](src/main.cpp#L99) to use an older ZED SDK version.

### Build the program

Open a terminal in the sample directory and execute the following command:

        mkdir build
        cd build
        cmake ..
        make -j8

We then need to make a symbolic link to the models folder to be able to loads it

        ln -s ~/path/to/openpose/models "$(pwd)"
        
A `models` folder should now be in the build folder


## Run the program

- Navigate to the build directory and launch the executable
- Or open a terminal in the build directory and run the sample :

        ./zed_openpose -net_resolution 656x368


<p align="center">
    <img src="./OpenPose_ZED.png", width="650">
</p>


## Options

Beyond the openpose option, several more were added, mainly:

Option             | Description
---------------------|------------------------------------
svo_path     | SVO file path to load instead of opening the ZED
ogl_ptcloud             | Boolean to show the point cloud in the OpenGL window
estimate_floor_plane       | Boolean to align the point cloud on the floor plane
opencv_display    | Enable the 2D View of OpenPose output
depth_display    | Display the depth map with OpenCV


Example :

        ./zed_openpose -net_resolution 320x240 -ogl_ptcloud true -svo_path ~/foo/bar.svo

## Notes

- This sample is a proof of concept and might not be robust to every situation, especially to detect the floor plane if the environment is cluttered.
- This sample was only tested on Linux but should be easy to run on Windows.
- This sample requires both Openpose and the ZED SDK which are heavily relying on the GPU.
- Only the body keypoints are currently used, however we could imagine doing the same for hand and facial keypoints, though the precision required might be a limiting factor.
