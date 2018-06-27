//
// Created by chaoz on 28/03/18.
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <string>
#include <boost/lexical_cast.hpp>

using namespace cv;
using namespace std;

int VideoHeight = 0;
int VideoWidth = 0;
cv_bridge::CvImagePtr cv_ptr;
Mat img;


void ImageReceived(const sensor_msgs::ImageConstPtr msg) {
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		img = cv_ptr->image.clone();
		cout<<"image received"<<endl;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}


int main(int argc, char** argv)
{
	ros::init(argc,argv,"recorder");
	ros::NodeHandle nh;
	string video_path;
	VideoCapture cap;
    ros::Subscriber image_sub;
	bool subscribe_from_topic = false;
	nh.getParam("subscribe_from_topic",subscribe_from_topic);
	string topic_name;
	nh.getParam("topic_name",topic_name);
	cout<<subscribe_from_topic<<endl;

	if(!subscribe_from_topic) {
		bool webcam_on = false;
		nh.getParam("webcam_on",webcam_on);
		if(!webcam_on) {
			ROS_ERROR("no image source is enable");
			exit(0);
		}
		int webcam_port = 0;
		nh.getParam("webcam_port",webcam_port);
		cap = VideoCapture(webcam_port);
		bool success=false;
		success = cap.read(img);
		if(!success) {
			ROS_ERROR_STREAM("fail to open webcam at port "<<webcam_port);
			exit(1);
		}
	}else {
		image_sub = nh.subscribe(topic_name,1,&ImageReceived);
		while(!cv_ptr){
			ros::spinOnce();
			}
	}


	nh.getParam("video_path",video_path);
	bool absolute_path = true;
	nh.getParam("absolute_path",absolute_path);
	if(!absolute_path) {

	    std::string pkg_path = ros::package::getPath("vision")+"/";
		video_path.insert(0,pkg_path);
	}
	ros::Rate rate_(30);
	VideoHeight=img.rows;
	VideoWidth=img.cols;
	VideoWriter video(video_path,CV_FOURCC('M','J','P','G'),30, Size(VideoWidth,VideoHeight));
	cout<<video_path<<endl;
	
	char key = 0;
	cout<<"start"<<endl;
    while(ros::ok()) {
		if(subscribe_from_topic) {
			ros::spinOnce();
		}else{
			bool success = false;
			success = cap.read(img);
			if(!success) break;
		}
        
        video.write(img);
        
		Mat out;
				
		rate_.sleep();
    }
	cout<<"end"<<endl;
	cap.release();
	video.release();
    	return 0;


}

