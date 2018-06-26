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
bool g_left_topic_active = false;
bool g_right_topic_active = false;
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
    if (!g_left_topic_active) {
		g_left_topic_active = true;
		ROS_INFO("Left camera is now active.");
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
    if (!g_right_topic_active) {
		g_right_topic_active = true;
		ROS_INFO("Right camera is now active.");
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
	
	ROS_INFO("Start configure subscriber.");
	ros::Subscriber left_image_sub, right_image_sub;
	left_image_sub = nh.subscribe(g_left_topic_name, 1, &LeftImageReceivedCB);
	right_image_sub = nh.subscribe(g_right_topic_name, 1, &RightImageReceivedCB);
	ROS_INFO("Wait for topics to be active.");
	ros::Rate r_(10);
	while(ros::ok() && !(g_right_topic_active && g_left_topic_active)) {
		ros::spinOnce();
		r_.sleep();
	}
	ROS_INFO("Done configure subscriber.");
	
	vector< Point3f > obj;
    for (int i = 0; i < g_chessboardSize.height; i++)
		for (int j = 0; j < g_chessboardSize.width; j++)
			obj.push_back(Point3f((float)j * g_calibrationSquareWidth, (float)i * g_calibrationSquareWidth, 0.0f));
			
	vector< vector< Point2f >> left_img_points, right_img_points;
	vector< vector< Point3f >> obj_points;
	
	
    camCal left_cam_cal(g_calibrationSquareWidth,g_chessboardSize);
    camCal right_cam_cal(g_calibrationSquareWidth,g_chessboardSize);


    vector<Mat> savedImages_left, savedImages_right;

    namedWindow("visual", WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL);
	r_ = ros::Rate(15);
    while (ros::ok()) {
        ros::spinOnce();
        
        vector<Point2f> left_foundPoints, right_foundPoints;
        bool found_left = false;
        bool found_right = false;
        
		Mat left_img_draw, right_img_draw, concat_img;
		//left
        found_left = findChessboardCorners(g_left_img, g_chessboardSize, left_foundPoints);
        g_left_img.copyTo(left_img_draw);

        if (!found_left) {
            continue;
        }
        drawChessboardCorners(left_img_draw, g_chessboardSize, left_foundPoints, found_left);
        
        //right
        found_right = findChessboardCorners(g_right_img, g_chessboardSize, right_foundPoints);
        g_right_img.copyTo(right_img_draw);

        if (!found_right) {
            continue;
        }
        drawChessboardCorners(right_img_draw, g_chessboardSize, right_foundPoints, found_right);
        
        hconcat(left_img_draw, right_img_draw, concat_img);
        system("clear");
        Mat left_camera_matrix = left_cam_cal.getCameraMatrix();
        Mat left_dist_coef = left_cam_cal.getDistanceCoefficients();
        Mat right_camera_matrix = right_cam_cal.getCameraMatrix();
        Mat right_dist_coef = right_cam_cal.getDistanceCoefficients();
        ROS_INFO_STREAM("Left:\n" << 
		                "    Camera Matrix:\n" <<
		                left_camera_matrix <<
		                "    Dist Coef:\n" <<
		                left_dist_coef);
		ROS_INFO_STREAM("Right:\n" << 
		                "    Camera Matrix:\n" <<
		                right_camera_matrix <<
		                "    Dist Coef:\n" <<
		                right_dist_coef);
        ROS_INFO(toString(savedImages.size())<<" images have been collected for each camera!");
        ROS_INFO("press enter 	>> start calibration.");
        ROS_INFO("press space 	>> save this image.");
        ROS_INFO("press s 		>> save calibration.");
        ROS_INFO("press esc 	>> exit.");
        ROS_INFO("press others 	>> skip this image.");
        ROS_INFO("wait 5 sec	>> skip this image.");
        imshow("visual", concat_img);
        char cKey = waitKey(5000);

        switch (cKey) {
            case 13:    // enter
                //start calibration
                if (savedImages_left.size() < 50) {
					ROS_WARN("More images required, try later");
                    break;
                }
                left_cam_cal.cameraCalibration(savedImages_left);
                right_cam_cal.cameraCalibration(savedImages_right);
                break;

            case 27:    // esc
                //exit
                return 0;
                break;

            case ' ':   // space
                //saving image
				Mat temp;
				g_left_img.copyTo(temp);
				savedImages_left.push_back(temp);
				g_right_img.copyTo(temp);
				savedImages_right.push_back(temp);
				cv::cornerSubPix(g_left_img, left_foundPoints, cv::Size(5, 5), cv::Size(-1, -1),
								 cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
				left_img_points.push_back(left_foundPoints);
				cv::cornerSubPix(g_right_img, right_foundPoints, cv::Size(5, 5), cv::Size(-1, -1),
								 cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
				right_img_points.push_back(right_foundPoints);
				obj_points.push_back(obj);
                break;
                
			case 's':	// s
				ROS_INFO("Start saving calibration results.");
				/**
				 * Implement
				 */
				break;

            default:	// others
                break;
        }
        r_.sleep();
    }

    return 0;
}
