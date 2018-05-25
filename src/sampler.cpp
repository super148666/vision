//
// Created by chaoz on 2/04/18.
//

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

#include <ros/ros.h>
#include <ros/package.h>

#include <iostream>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <boost/lexical_cast.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

int numCones = 0;
int numNonCones = 0;
string pathCones = "./cones/";
string pathNonCones = "./non-cones/";
string video_filename = ".";

int sampleSize = 64;
int windowSize = 64;
double scale_x = 0.5;
double scale_y = 0.5;
int VideoWidth = 160;
int VideoHeight = 120;
int ProcessWidth = 160;
int ProcessHeight = 120;

Mat img,imgshow,imgmouse,sample;

bool newSample = false;

vector<string> emptyCones;
vector<string> emptyNonCones;


std::vector<cv::Rect> get_sliding_windows(Rect roi,int winWidth,int winHeight, int step)
{
    std::vector<cv::Rect> rects;

    for(int i=roi.y;i<(roi.y+roi.height);i+=step)
    {
        if((i+winHeight)>(roi.y+roi.height)){break;}
        for(int j=roi.x;j< (roi.x+roi.width);j+=step)
        {
            if((j+winWidth)>(roi.x+roi.width)){break;}
            cv::Rect rect(j,i,winWidth,winHeight);
            rects.push_back(rect);
        }
    }
    return rects;
}

void CollectSamples()
{
    int index = 0;
    string filename = "not saved";
    char key = waitKey(10);
    while(1) {
        if(newSample==true)
        {
            newSample=false;
            filename="not saved";
            ROS_INFO_STREAM("current index: "<<index<<" | "<<filename);
            ROS_INFO_STREAM("a:- | d:+");
            ROS_INFO_STREAM("w:cone | s:non-cone | e:redo | space:next frame");

        }
        switch(key){
            case 'a':
                windowSize-=2;
                if(windowSize<2) windowSize=2;
                break;

            case 'd':
                windowSize+=2;
                if(windowSize>ProcessHeight) windowSize=ProcessHeight;
                if(windowSize>ProcessWidth) windowSize=ProcessWidth;
                break;

            case 'w':   //cones
                if(filename=="not saved") {
                    if(emptyCones.empty())
                        filename = pathCones + "image" + boost::lexical_cast<string>(++numCones) + ".png";
                    else {
                        filename=emptyCones.front();
                        emptyCones.erase(emptyCones.begin());
                    }
                    imwrite(filename, sample);
                }
                else {
                    if(filename.find("non-")!=string::npos) {
                        emptyNonCones.push_back(filename);
                        if(emptyCones.empty())
                            filename = pathCones + "image" + boost::lexical_cast<string>(++numCones) + ".png";
                        else {
                            filename=emptyCones.front();
                            emptyCones.erase(emptyCones.begin());
                        }
                        imwrite(filename, sample);
                    }
                }
                ROS_INFO_STREAM("image "<<index<<" has been saved at "<<filename);
                index++;
                break;

            case 's':   //non-cones
                if(filename=="not saved") {
                    if(emptyNonCones.empty())
                        filename = pathNonCones+"image" + boost::lexical_cast<string>(++numNonCones) + ".png";
                    else {
                        filename=emptyNonCones.front();
                        emptyNonCones.erase(emptyNonCones.begin());
                    }
                    imwrite(filename, sample);
                }
                else {
                    if(filename.find("non-")==string::npos) {
                        emptyCones.push_back(filename);
                        if(emptyNonCones.empty())
                            filename = pathNonCones+"image" + boost::lexical_cast<string>(++numNonCones) + ".png";
                        else {
                            filename=emptyNonCones.front();
                            emptyNonCones.erase(emptyNonCones.begin());
                        }
                        imwrite(filename, sample);
                    }
                }
                ROS_INFO_STREAM("image "<<index<<" has been saved at "<<filename);
                index++;
                break;

            case 'e':   //delete
                if(filename=="not saved") {
                    ROS_INFO_STREAM("image "<<index<<" was not saved");
                }
                else {
                    string command = "rm -rf "+filename;
                    system(command.c_str());
                    if(filename.find("non-")==string::npos)
                        emptyCones.push_back(filename);
                    else
                        emptyNonCones.push_back(filename);
                    filename = "not saved";
                }
                index--;
                break;

            case ' ':
                destroyWindow("sample");
                return;

            default:
                break;
        }
        key = waitKey(10);
    }
}

void MouseCB(int event, int x, int y, int flags, void* ptr)
{
    if(event==EVENT_LBUTTONDOWN)
    {
        int Px = x - windowSize/2;
        int Py = y - windowSize/2;
        if(Px<0) Px=0;
        if(Py<0) Py=0;
        if(Px>ProcessHeight-windowSize) Px = ProcessHeight-windowSize;
        if(Py>ProcessWidth-windowSize) Py = ProcessWidth-windowSize;
        imgshow=img.clone();
        Rect window(Px,Py,windowSize,windowSize);
        rectangle(imgshow,window,Scalar(0,0,200));
        imshow("img",imgshow);
        sample = img(window).clone();
        resize(sample,sample,Size(sampleSize,sampleSize));
        destroyWindow("sample");
        namedWindow("sample",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
        moveWindow("sample",10,10);
        imshow("sample",sample);
        newSample = true;
    }
    if(event==EVENT_MOUSEMOVE)
    {
        int Px = x - windowSize/2;
        int Py = y - windowSize/2;
        if(Px<0) Px=0;
        if(Py<0) Py=0;
        if(Px>ProcessHeight-windowSize) Px = ProcessHeight-windowSize;
        if(Py>ProcessWidth-windowSize) Py = ProcessWidth-windowSize;
        imgmouse=imgshow.clone();
        Rect window(Px,Py,windowSize,windowSize);
        rectangle(imgmouse,window,Scalar(200,0,0));
        imshow("img",imgmouse);
    }
    if(event==EVENT_RBUTTONUP)
    {
        exit(0);
    }
}

void InitGlobalVariables(ros::NodeHandle nh) {
    std::string pkg_path = ros::package::getPath("vision")+"/";

    nh.getParam("num_of_cones",numCones);
    nh.getParam("num_of_noncones",numNonCones);
    nh.getParam("path_of_cones",pathCones);
    nh.getParam("path_of_noncones",pathNonCones);
    nh.getParam("scale_x",scale_x);
    nh.getParam("scale_y",scale_y);
    nh.getParam("video_filename",video_filename);
    nh.getParam("sample_size",sampleSize);

    windowSize = sampleSize;

    video_filename.insert(0,pkg_path);
    pathCones.insert(0,pkg_path);
    pathNonCones.insert(0,pkg_path);

    ROS_INFO("configure done");


}

int main(int argc, char** argv)
{
    ros::init(argc,argv,"sampler");
    ros::NodeHandle nh;
    InitGlobalVariables(nh);
    auto cap = VideoCapture(video_filename);
    int count = 0;
    namedWindow("img",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
    setMouseCallback("img",MouseCB);
    bool first = true;
    while(cap.isOpened()) {
        system("clear");
        ROS_INFO_STREAM("frame no:"<< ++count);
        ROS_INFO_STREAM("a:- | d:+");
        ROS_INFO_STREAM("w:cone | s:non-cone | e:redo | space:next frame");
        std::vector<cv::Point> Locations;
        bool success = cap.read(img);
        if(!success) break;
        if(first) {
            VideoHeight = img.cols;
            VideoWidth = img.rows;
            ProcessHeight = VideoHeight*scale_y;
            ProcessWidth = VideoWidth*scale_x;
        }
        resize(img,img,Size(ProcessHeight,ProcessWidth));
        imgshow=img.clone();
        imshow("img",imgshow);
        waitKey(1);
        sample = Mat(windowSize,windowSize,CV_8UC3,Scalar(0,0,0));
        CollectSamples();
        imgshow.release();
        img.release();
    }
    return 0;

}

