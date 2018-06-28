#include <iostream>

#include <string>
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// default parameters
string g_calSavePath = "default.yml";
string g_left_topic_name = "/left/image";
string g_right_topic_name = "/right/image";
bool g_left_topic_active = false;
bool g_right_topic_active = false;
Mat g_left_img, g_right_img;
Mat Kl,Kr,Dl,Dr,Rl,Rr,Pl,Pr;
Size g_img_size;

void LeftImageReceivedCB(const sensor_msgs::ImageConstPtr msg) {
    try
    {
		cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
		g_left_img = cv_ptr->image.clone();
        cvtColor(g_left_img, g_left_img, cv::COLOR_RGB2GRAY);
		//cout<<"left image received"<<endl;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    if (!g_left_topic_active) {
		g_left_topic_active = true;
		g_img_size = g_left_img.size();
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
		//cout<<"right image received"<<endl;
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

void LoadCalibrationMatrix(string filename) {
	ROS_INFO("Start load calibration matrixs.");
	FileStorage fs(filename, FileStorage::READ);
	fs["Kl"] >> Kl;
	fs["Kr"] >> Kr;
	fs["Dl"] >> Dl;
	fs["Dr"] >> Dr;
	fs["Rl"] >> Rl;
	fs["Rr"] >> Rr;
	fs["Pl"] >> Pl;
	fs["Pr"] >> Pr;
	fs.release();
	
	ROS_INFO("Done load calibration matrixs.");
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
	
	
	// load left image topic name
	param_name = "left_topic_name";
	param_string_ptr = &g_left_topic_name;
    if (!nh.getParam(param_name, param_string)) {
		ROS_WARN_STREAM("Missing parameters " << param_name << ".");
		ROS_WARN_STREAM("Default setting will be used for " << param_name);
	} else *param_string_ptr = param_string;
	if (param_string_ptr->empty()) {
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
	if (param_string_ptr->empty()) {
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
	if (param_string_ptr->empty()) {
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
	ROS_INFO_STREAM("Setting " << param_name << " to " << ((*param_bool_ptr)?"true":"false") << ".");
    
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
	
	LoadCalibrationMatrix(g_calSavePath);
	Mat left_map_1, left_map_2, right_map_1, right_map_2;
	initUndistortRectifyMap(Kl, Dl, Rl, Pl, g_img_size,  CV_16SC2, left_map_1, left_map_2);
	initUndistortRectifyMap(Kr, Dr, Rr, Pr, g_img_size,  CV_16SC2, right_map_1, right_map_2);
	
	namedWindow("visual", WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL);
	r_ = ros::Rate(15);

	Ptr<StereoBM> sbm = StereoBM::create(16,9);
	
    sbm->setDisp12MaxDiff(1);
    sbm->setSpeckleRange(9);
    sbm->setSpeckleWindowSize(2);
    sbm->setUniquenessRatio(1);
    sbm->setTextureThreshold(1999);
    sbm->setMinDisparity(-16);
    sbm->setPreFilterCap(63);
    sbm->setPreFilterSize(99);



    while (ros::ok()) {
        ros::spinOnce();
        Mat left_img_undistorted, right_img_undistorted, depth_img, concat_img_undistorted, show_img;
        remap(g_left_img, left_img_undistorted, left_map_1, left_map_2, INTER_LINEAR);
        remap(g_right_img, right_img_undistorted, right_map_1, right_map_2, INTER_LINEAR);
		sbm->compute(left_img_undistorted, right_img_undistorted, depth_img);
		normalize(depth_img, depth_img, 0, 255, CV_MINMAX, CV_8U);
        hconcat(left_img_undistorted, right_img_undistorted, concat_img_undistorted);
        hconcat(concat_img_undistorted,depth_img, show_img);
        imshow("visual", show_img);
        waitKey(1);
        r_.sleep();
    }

    return 0;
}
