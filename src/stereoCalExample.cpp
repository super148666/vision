#include <iostream>

#include <string>
#include "camCal.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// default chessboard parameters
float g_calibrationSquareWidth = 0.03508f;
Size g_chessboardSize = Size(8, 6);
string g_calSavePath = "default.yml";
string g_left_topic_name = "/left/image";
string g_right_topic_name = "/right/image";

Mat g_left_img, g_right_img;

void LeftImageReceivedCB(const sensor_msgs::ImageConstPtr msg) {
    try
    {
		cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
		g_left_img = cv_ptr->image.clone();
        cvtColor(g_left_img, g_left_img, cv::COLOR_RGB2GRAY);
		cout<<"left image received"<<endl;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}

void RightImageReceivedCB(const sensor_msgs::ImageConstPtr msg) {
    try
    {
		cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
		g_right_img = cv_ptr->image.clone();
        cvtColor(g_right_img, g_right_img, cv::COLOR_RGB2GRAY);
		cout<<"right image received"<<endl;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}

void ReadParams(ros::NodeHandle nh) {
	ROS_INFO("Start read parameters.");
	string param_name;
	int param_int;
	int* param_int_ptr;
	float param_float;
	float* param_float_ptr;
	string param_string;
	string* param_string_ptr;
	bool param_bool;
	bool* param_bool_ptr;
	bool error_exit = false;
	
	// get pkg root path
    std::string pkg_path = ros::package::getPath("vision")+"/";
	
	// load parameters
	int c_width, c_height;
	c_width = 9;	// default
	c_height = 7;	// default
	
	// load chessboard width
	param_name = "chessboard_width";
	param_int_ptr = &c_width;
    if (!nh.getParam(param_name, param_int)) {
		ROS_WARN_STREAM("Missing parameters " << param_name << ".");
		ROS_WARN_STREAM("Default setting will be used for " << param_name);
	} else *param_int_ptr = param_int;
	// check chessboard width
	if (*param_int_ptr <= 1) {
		ROS_ERROR_STREAM("Invalid " << param_name << " as " << *param_int_ptr << ".");
		ROS_ERROR_STREAM(param_name << " must be larger than 1.");
		error_exit = true;
	}
	ROS_INFO_STREAM("Setting " << param_name << " to " << *param_int_ptr << ".");
	
	// load chessboard height
	param_name = "chessboard_height";
	param_int_ptr = &c_height;
    if (!nh.getParam(param_name, param_int)) {
		ROS_WARN_STREAM("Missing parameters " << param_name << ".");
		ROS_WARN_STREAM("Default setting will be used for " << param_name);
	} else *param_int_ptr = param_int;
	// check chessboard width
	if (*param_int_ptr <= 1) {
		ROS_ERROR_STREAM("Invalid " << param_name << " as " << *param_int_ptr << ".");
		ROS_ERROR_STREAM(param_name << " must be larger than 1.");
		error_exit = true;
	}
	ROS_INFO_STREAM("Setting " << param_name << " to " << *param_int_ptr << ".");
		        
    // make sure width is larger than height
    if (c_width > c_height) {
		int temp = c_height;
		c_height = c_width;
		c_width = temp;
	}
	// define size using width and height
    g_chessboardSize = Size(c_width-1, c_height-1);
    
    // load chessboard_square_width
	param_name = "chessboard_square_width";
	param_float_ptr = &g_calibrationSquareWidth;
    if (!nh.getParam(param_name, param_float)) {
		ROS_WARN_STREAM("Missing parameters " << param_name << ".");
		ROS_WARN_STREAM("Default setting will be used for " << param_name);
	} else *param_float_ptr = param_float;
	if (*param_float_ptr <= 0) {
		ROS_ERROR_STREAM("Invalid " << param_name << " as " << *param_float_ptr << ".");
		ROS_ERROR_STREAM(param_name << " must be larger than 0.");
		error_exit = true;
	}
	ROS_INFO_STREAM("Setting " << param_name << " to " << *param_float_ptr << ".");
	
	// load left image topic name
	param_name = "left_topic_name";
	param_string_ptr = &g_left_topic_name;
    if (!nh.getParam(param_name, param_string)) {
		ROS_WARN_STREAM("Missing parameters " << param_name << ".");
		ROS_WARN_STREAM("Default setting will be used for " << param_name);
	} else *param_string_ptr = param_string;
	if (*param_string_ptr.empty()) {
		ROS_ERROR_STREAM("Invalid empty " << param_name << ".");
		error_exit = true;
	}
	ROS_INFO_STREAM("Setting " << param_name << " to " << *param_string_ptr << ".");
	
	// load right image topic name
	param_name = "right_topic_name";
	param_string_ptr = &g_right_topic_name;
    if (!nh.getParam(param_name, param_string)) {
		ROS_WARN_STREAM("Missing parameters " << param_name << ".");
		ROS_WARN_STREAM("Default setting will be used for " << param_name);
	} else *param_string_ptr = param_string;
	if (*param_string_ptr.empty()) {
		ROS_ERROR_STREAM("Invalid empty " << param_name << ".");
		error_exit = true;
	}
	ROS_INFO_STREAM("Setting " << param_name << " to " << *param_string_ptr << ".");
	
    
    // load save_path
	param_name = "save_path";
	param_string_ptr = &g_calSavePath;
    if (!nh.getParam(param_name, param_string)) {
		ROS_WARN_STREAM("Missing parameters " << param_name << ".");
		ROS_WARN_STREAM("Default setting will be used for " << param_name);
	} else *param_string_ptr = param_string;
	if (*param_string_ptr.empty()) {
		ROS_ERROR_STREAM("Invalid empty " << param_name << ".");
		error_exit = true;
	}
	ROS_INFO_STREAM("Setting " << param_name << " to " << *param_string_ptr << ".");
	
    
    bool absolute_path = false;
    param_name = "absolute_path";
    param_bool_ptr = &absolute_path;
    if (!nh.getParam(param_name, param_bool)) {
		ROS_WARN_STREAM("Missing parameters " << param_name << ".");
		ROS_WARN_STREAM("Default setting will be used for " << param_name);
	} else *param_bool_ptr = param_bool;
	ROS_INFO_STREAM("Setting " << param_name << " to " << *param_bool_ptr?"true":"false" << ".");
    
    if(!absolute_path) {
        g_calSavePath.insert(0,pkg_path);
    }
    
    if (error_exit) {
		ROS_ERROR("Error during reading parameters.");
		ROS_ERROR("Aborting...");
		exit(-1);
	}

    ROS_INFO("Done read parameters.");


}

int main(int argc, char **argv) {
	ros::init(argc, argv, "stereo_calibration");
	ros::NodeHandle nh;
	ReadParams(nh);
	ros::Subscriber left_image_sub, right_image_sub;
	left_image_sub = nh.subscribe(
    camCal leftCal(g_calibrationSquareWidth,g_chessboardSize);
    camCal rightCal(g_calibrationSquareWidth,g_chessboardSize);
    int fps = 10;
    

    Mat cameraMatrix_left = Mat::eye(3, 3, CV_64F);
    Mat distanceCoefficients_left;
	Mat cameraMatrix_right = Mat::eye(3, 3, CV_64F);
    Mat distanceCoefficients_right;

    vector<Mat> savedImages_left, savedImages_right;

    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    namedWindow("visual", WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL);

    while (ros::ok()) {
        ros::spinOnce();
        vector<Point2f> foundPoints;
        bool found = false;
		Mat left_img_draw, right_img_draw, concat_img;
		//left
        found = findChessboardCorners(left_img, g_chessboardSize, foundPoints);
        left_img.copyTo(left_img_draw);

        if (found) {
            drawChessboardCorners(drawToFrame, g_chessboardSize, foundPoints, found);
        } else continue;

        imshow("visual", drawToFrame);
        char cKey = waitKey(1000 / 60);
        system("clear");
        cout<<" "<<savedImages.size()<<" have been collected!\n";

        switch (cKey) {
            case 13:    // enter
                //start calibration
                if (savedImages.size() < 50) {
                    cout << "More images required, try later" << endl;
                    break;
                }
                camCal1.cameraCalibration(savedImages);
                camCal1.saveCameraCalibration("camera_calibration_results");
                break;

            case 27:    // esc
                //exit
                return 0;
                break;

            case ' ':   // space
                //saving image
                if (found) {
                    Mat temp;
                    frame.copyTo(temp);
                    savedImages.push_back(temp);
                }
                break;

            default:
                break;
        }
    }

    return 0;
}
