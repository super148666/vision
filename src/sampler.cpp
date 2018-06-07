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


using namespace cv;
using namespace std;

#define MODE_SINGLE 0
#define MODE_BATCH 1

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
int win_stride = 8;

Mat img,imgshow,imgmouse,sample;
vector<Mat> samples;
vector<string> filenames;

string default_filename = "not saved";

bool newSample = false;

vector<string> emptyCones;
vector<string> emptyNonCones;

int frame_no = 0;

bool samplesLock = false;

// void LockSamplesLock()
// {
//     while(samplesLock) {
//         waitKey(10);
//     }
//     samplesLock = true;
// }

void UnlockSamplesLock() {
    samplesLock = false;
}

void CollectSamples()
{
    
    int index = 0;
    char key = 0;
    int batch_size = 0;
    
    while(1) {
        
        // update terminal
        // LockSamplesLock();
        if(newSample==true)
        {
            // system("clear");
            batch_size = samples.size();
            
            // ROS_INFO_STREAM("vector size of filenames: "<<filenames.size());
            
            newSample=false;

        }

        switch(key){

            case 'w':   //cones
                if(filenames.front()==default_filename) {
                    for(int i = 0; i < batch_size; i++) {
                        if(emptyCones.empty()) {
                            filenames[i] = pathCones + "image" + boost::lexical_cast<string>(++numCones) + ".png";
                        }
                        else {
                            filenames[i]=emptyCones.front();
                            emptyCones.erase(emptyCones.begin());
                        }
                        imwrite(filenames[i], samples[i]);
                    }
                }
                else if(filenames.front().find(pathNonCones)!=string::npos) {
                    for(int i = 0; i < batch_size; i++) {
                        emptyNonCones.push_back(filenames[i]);
                        if(emptyCones.empty()) {
                            filenames[i] = pathCones + "image" + boost::lexical_cast<string>(++numCones) + ".png";
                        }
                        else {
                            filenames[i]=emptyCones.front();
                            emptyCones.erase(emptyCones.begin());
                        }
                        imwrite(filenames[i], samples[i]);
                    }
                        
                }
                ROS_INFO_STREAM("image "<<index<<" has been saved at "<<filenames.front());
                index++;
                break;

            case 's':   //non-cones
                if(filenames.front()==default_filename) {
                    for(int i = 0; i < batch_size; i++) {
                        if(emptyNonCones.empty())
                            filenames[i] = pathNonCones+"image" + boost::lexical_cast<string>(++numNonCones) + ".png";
                        else {
                            filenames[i]=emptyNonCones.front();
                            emptyNonCones.erase(emptyNonCones.begin());
                        }
                        imwrite(filenames[i], samples[i]);
                    }
                }
                else if(filenames.front().find(pathCones)!=string::npos) {
                    for(int i = 0; i < batch_size; i++) {
                        emptyCones.push_back(filenames[i]);
                        if(emptyNonCones.empty())
                            filenames[i] = pathNonCones+"image" + boost::lexical_cast<string>(++numNonCones) + ".png";
                        else {
                            filenames[i]=emptyNonCones.front();
                            emptyNonCones.erase(emptyNonCones.begin());
                        }
                        imwrite(filenames[i], samples[i]);
                    }
                }
                ROS_INFO_STREAM("image "<<index<<" has been saved at "<<filenames.front());
                index++;
                break;

            case 'e':   //delete
                if(filenames.front()==default_filename) {
                    ROS_INFO_STREAM("image "<<index<<" was not saved");
                }
                else {
                    for(int i = 0; i < batch_size; i++) {
                        string command = "rm -rf "+filenames[i];
                        system(command.c_str());
                        if(filenames[i].find(pathNonCones)!=string::npos)
                            emptyNonCones.push_back(filenames[i]);
                        else if(filenames[i].find(pathCones)!=string::npos)
                            emptyCones.push_back(filenames[i]);
                        filenames[i] = default_filename;
                    }
                    index--;
                }
                
                break;

            case ' ':
                destroyWindow("sample");
                samples.clear();
                return;

            default:
                // newAction = false;
                break;
        }
        key = waitKey(10);
    }
}

vector<Rect> get_sliding_windows(Rect roi, int winWidth, int winHeight, int step) {
    vector<Rect> rects;
    for (int i = roi.y; i < (roi.y + roi.height); i += step) {
        if ((i + winHeight) > (roi.y + roi.height)) { break; }
        for (int j = roi.x; j < (roi.x + roi.width); j += step) {
            if ((j + winWidth) > (roi.x + roi.width)) { break; }
            cv::Rect rect(j, i, winWidth, winHeight);
            rects.push_back(rect);
        }
    }
    return rects;
}

void UpdateTerminal()
{
    int batch_size = samples.size();
    system("clear");
    ROS_INFO_STREAM("frame no:"<< frame_no);
    ROS_INFO_STREAM("num of images: "<<batch_size);
    if(batch_size!=0) {
        if(filenames.front().find(pathCones)!=string::npos) {
            ROS_INFO("-----CONES-----");
        } else if(filenames.front().find(pathNonCones)!=string::npos) {
            ROS_INFO("-----NON-CONES-----");
        } else {
            ROS_INFO("-----PENDING-----");
        }
        if(batch_size==1) {
            ROS_INFO_STREAM("location: "<<filenames.front());
        } else {
            ROS_INFO_STREAM("first location: "<<filenames.front());
            ROS_INFO_STREAM("last location: "<<filenames.back());
        }
    } else {
        ROS_WARN("no available images");
    }
    ROS_INFO_STREAM("w:cone | s:non-cone | e:redo | space:next frame");
    ROS_INFO_STREAM("total cones: "<<numCones<<" non-cones: "<<numNonCones);
}

void MouseCB(int event, int x, int y, int flags, void* ptr)
{
    static bool first_right_click = false;
    static Point tl_point(0,0);
    static Point br_point(0,0);
    // CTRL + Left Click will terminate the program
    if(flags==(EVENT_FLAG_CTRLKEY+EVENT_FLAG_LBUTTON))
    {
        exit(0);
    }

    // Only Left Click: sample single image
    if(event==EVENT_LBUTTONDOWN)
    {
        if(first_right_click) {
            first_right_click = false;
            imshow("img",img);
            return;
        }

        samples.clear();
        filenames.clear();
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
        samples.push_back(sample);
        filenames.push_back(default_filename);
        imshow("sample",sample);
        newSample = true;
    }

    // Moving Mouse: show window frame
    if(event==EVENT_MOUSEMOVE)
    {
        int Px, Py;        
        imgmouse=imgshow.clone();
        Rect window;       
        Px = x - windowSize/2;
        Py = y - windowSize/2;
        if(Px<0) Px=0;
        if(Py<0) Py=0;
        if(Px>ProcessHeight-windowSize) Px = ProcessHeight-windowSize;
        if(Py>ProcessWidth-windowSize) Py = ProcessWidth-windowSize;
        window = Rect(Px,Py,windowSize,windowSize);
        rectangle(imgmouse,window,Scalar(200,0,0));
        if(first_right_click) {
            Px = x + windowSize/2;
            Py = y + windowSize/2;
            if(Px<(tl_point.x+windowSize)) Px=(tl_point.x+windowSize);
            if(Py<(tl_point.y+windowSize)) Py=(tl_point.y+windowSize);
            if(Px>ProcessHeight) Px = ProcessHeight;
            if(Py>ProcessWidth) Py = ProcessWidth;
            window = Rect(tl_point, Point(Px,Py));
            rectangle(imgmouse,window,Scalar(200,0,0));
        }
        circle(imgmouse,Point(Px,Py),2,Scalar(200,0,0),-1);
        line(imgmouse, Point(Px,0), Point(Px,ProcessWidth),Scalar(200,0,0),1);
        line(imgmouse,Point(0,Py),Point(ProcessHeight,Py),Scalar(200,0,0),1);
        imshow("img",imgmouse);
    }

    // Right Click: two right clicks select a region to sample batch of images
    if(event==EVENT_RBUTTONDOWN)
    {
        int Px, Py;
        if( first_right_click == false ) {
            Px = x - windowSize/2;
            Py = y - windowSize/2;
            if(Px<0) Px=0;
            if(Py<0) Py=0;
            if(Px>ProcessHeight-windowSize) Px = ProcessHeight-windowSize;
            if(Py>ProcessWidth-windowSize) Py = ProcessWidth-windowSize;
            destroyWindow("sample");
            imgshow=img.clone();
            tl_point = Point(Px,Py);
            first_right_click = true;
        } else {
            Px = x + windowSize/2;
            Py = y + windowSize/2;
            if(Px<(tl_point.x+windowSize)) Px=(tl_point.x+windowSize);
            if(Py<(tl_point.y+windowSize)) Py=(tl_point.y+windowSize);
            if(Px>ProcessHeight) Px = ProcessHeight;
            if(Py>ProcessWidth) Py = ProcessWidth;
            br_point = Point(Px,Py);
            Rect sampling_roi(tl_point, br_point);
            sample = img(sampling_roi).clone();
            namedWindow("sample",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
            moveWindow("sample",10,10);
            imshow("sample", sample);
            vector<Rect> windows = get_sliding_windows(sampling_roi, windowSize, windowSize, win_stride);
            
            samples.clear();
            filenames.clear();
            for(int i = 0; i < windows.size(); i++) {
                samples.push_back(img(windows[i]).clone());
                filenames.push_back(default_filename);
            }
            first_right_click = false;
            newSample=true;
        }
    }

    UpdateTerminal();
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
    nh.getParam("sample_stride",win_stride);

    windowSize = sampleSize;
    bool absolute_path = true;
    nh.getParam("absolute_path",absolute_path);
    if(!absolute_path) {
        video_filename.insert(0,pkg_path);
        pathCones.insert(0,pkg_path);
        pathNonCones.insert(0,pkg_path);
    }
    samples.clear();
    emptyCones.clear();
    emptyNonCones.clear();

    ROS_INFO("configure done");


}

int main(int argc, char** argv)
{
    ros::init(argc,argv,"sampler");
    ros::NodeHandle nh;
    InitGlobalVariables(nh);
    auto cap = VideoCapture(video_filename);
    namedWindow("img",WINDOW_AUTOSIZE|WINDOW_GUI_NORMAL);
    setMouseCallback("img",MouseCB);
    bool first = true;
    while(cap.isOpened()&&ros::ok()) {
        
        
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
        // waitKey(1);
        sample = Mat(windowSize,windowSize,CV_8UC3,Scalar(0,0,0));
        CollectSamples();
        imgshow.release();
        img.release();
        frame_no++;
    }
    return 0;

}

