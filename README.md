# vision
Personal vision tool box
including sampling / labeling / calibration and ...
## Setup
### Install
```
cd
mkdir -p vision_ws/src
cd vision_ws/src
catkin_init_workspace
git clone https://github.com/super148666/vision.git
cd ..
catkin_make
source devel/setup.bash
```
# Stereo Calibration
## Config
```
cd ~/vision_ws/src/vision/param/
gedit stereo_calib.yaml
```
Following parameters are configurable:
* chessboard_width: number of cells along the longer edge
* chessboard_height: number of cells along the shorter edge
* chessboard_square_width: width for single cell on chessboard in meter
* left_topic_name: topic name in string for letf camera
* right_topic_name: topic name in string for right camera
* absolute_path: true if the save path should be absolute, otherwise false
* save_path: path for storing calibration outcome, as string

## Usage
```
source ~/vision_ws/devel/setup.bash
roslaunch vision stereo_calib.launch
```
