//
// Created by chaoz on 28/03/18.
//

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
/*
void MouseCB1(int event, int x, int y, int flags, void* ptr)
{
	static int count = 0;
	if(event==EVENT_RBUTTONDOWN)
	{
		destroyWindow("transformed");
		system("clear");
		count = 0;
	}
	
    if(event==EVENT_LBUTTONDOWN)
    {
		if(count>=4) {
			Mat h = getPerspectiveTransform(pts_dst1,pts_src1);
			Mat out;
			warpPerspective(img1, out, h, img1.size());
			namedWindow("transformed",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
			imshow("transformed",out);
			return;
		}
		count++;
        ROS_INFO_STREAM("video1 x:"<<x<<" y:"<<y); 
        pts_dst1.push_back(Point2f(x,y));
    }
    
    if(event==EVENT_MOUSEMOVE)
    {
        int Px = x - windowSize/2;
        int Py = y - windowSize/2;
        if(Px<0) Px=0;
        if(Py<0) Py=0;
        if(Px>VideoHeight1-windowSize) Px = VideoHeight1-windowSize;
        if(Py>VideoWidth1-windowSize) Py = VideoWidth1-windowSize;
        Rect window(Px,Py,windowSize,windowSize);
        Mat imgshow = img1.clone();
        circle(imgshow, Point(x,y), 3, Scalar(0,255,0));
        Mat sample = imgshow(window).clone();
        destroyWindow("sample");
        namedWindow("sample",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
        moveWindow("sample",10,10);
        imshow("sample",sample);
    }
}

void MouseCB2(int event, int x, int y, int flags, void* ptr)
{
	static int count = 0;
	if(event==EVENT_RBUTTONDOWN)
	{
		destroyWindow("transformed");
		system("clear");
		
	}
    if(event==EVENT_LBUTTONDOWN)
    {
		count++;
        ROS_INFO_STREAM("video2 x:"<<x<<" y:"<<y);
        
    }
    if(event==EVENT_MOUSEMOVE)
    {
        int Px = x - windowSize/2;
        int Py = y - windowSize/2;
        if(Px<0) Px=0;
        if(Py<0) Py=0;
        if(Px>VideoHeight2-windowSize) Px = VideoHeight2-windowSize;
        if(Py>VideoWidth2-windowSize) Py = VideoWidth2-windowSize;
        Rect window(Px,Py,windowSize,windowSize);
        Mat imgshow = img1.clone();
        circle(imgshow, Point(x,y), 3, Scalar(0,255,0));
        Mat sample = img2(window).clone();
        destroyWindow("sample");
        namedWindow("sample",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
        moveWindow("sample",10,10);
        imshow("sample",sample);
    }
}
*/

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
	/*
	
	pts_src1.push_back(Point2f(0+VideoWidth1/3,0+VideoHeight1/3));
	pts_src1.push_back(Point2f(VideoWidth1-VideoWidth1/3,0+VideoHeight1/3));
	pts_src1.push_back(Point2f(VideoWidth1-VideoWidth1/3,VideoHeight1-VideoHeight1/3));
	pts_src1.push_back(Point2f(0+VideoWidth1/3,VideoHeight1-VideoHeight1/3));
	
	pts_src1.push_back(Point2f(0+VideoHeight1*4/10,0+VideoWidth1*4/10));
	pts_src1.push_back(Point2f(VideoHeight1-VideoHeight1*4/10,0+VideoWidth1*4/10));
	pts_src1.push_back(Point2f(VideoHeight1-VideoHeight1*4/10,VideoWidth1-VideoWidth1*4/10));
	pts_src1.push_back(Point2f(0+VideoHeight1*4  /10,VideoWidth1-VideoWidth1*4/10));
	
	pts_dst1.push_back(Point2f(235,320));
	pts_dst1.push_back(Point2f(435,322));
	pts_dst1.push_back(Point2f(448,345));
	pts_dst1.push_back(Point2f(225,342));
	Mat h = getPerspectiveTransform(pts_dst1,pts_src1);
	
	pts_src2.push_back(Point2f(0,0));
	pts_src2.push_back(Point2f(VideoWidth2,0));
	pts_src2.push_back(Point2f(VideoWidth2,VideoHeight2));
	pts_src2.push_back(Point2f(0,VideoHeight2));
	
	
	
		int frame_width = cap1.get(CV_CAP_PROP_FRAME_WIDTH);
		int frame_height = cap1.get(CV_CAP_PROP_FRAME_HEIGHT);
		VideoWriter video1(o1,CV_FOURCC('M','J','P','G'),30, Size(frame_width,frame_height),true);
		
		frame_width = cap2.get(CV_CAP_PROP_FRAME_WIDTH);
		frame_height = cap2.get(CV_CAP_PROP_FRAME_HEIGHT);
		VideoWriter video2(o2,CV_FOURCC('M','J','P','G'),30, Size(frame_width,frame_height),true);
	
	
	namedWindow("video1",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
	namedWindow("video2",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
	namedWindow("transformed",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
	//setMouseCallback("video1",MouseCB1);
	//setMouseCallback("video2",MouseCB2);
	*/
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
        // VideoHeight=img.cols;
        // VideoWidth=img.rows;
        video.write(img);
        
		Mat out;
		// warpPerspective(img1, out, h, img1.size());
		
		// imshow("transformed",out);
		// moveWindow("transformed",100,100);
		// imshow("video1",img1);
		//imshow("video2",img2);
		// key = waitKey(20);
		
		rate_.sleep();
    }
	cout<<"end"<<endl;
	cap.release();
	video.release();
    	return 0;


}

