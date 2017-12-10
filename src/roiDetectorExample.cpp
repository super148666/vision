//
// Created by chaoz on 6/12/17.
//

#include <iostream>
#include <strstream>
#include <fstream>
#include "roiDetector.h"

#define HISTROY 3
#define VAR_THRES 0
#define DETECT_SHADOWS false

int main(int argv, char** argc) {

    roiDetector roiDetector1(HISTROY,VAR_THRES,DETECT_SHADOWS);

    cv::VideoCapture vid;
//    vid.open("../project_video.mp4");
//    vid.open("../challenge_video.mp4");
//    vid.open("../harder_challenge_video.mp4");

    vid.open("../section0.mov");

//    vid.open(0);

    int fps = 60;
    int period = 1000/fps;
//    int period = 1;
    if(!vid.isOpened()) {
        std::cout<<"Fail to open webcam."<<std::endl;
        return 1;
    }
    while(cv::waitKey(period)!=' ') {
        cv::Mat frame;
        if(!vid.read(frame)) break;
        cv::resize(frame,frame,cv::Size(frame.cols/1,frame.rows/1));
        int count = roiDetector1.detect(&frame, true);
        cv::imshow("draw to frame",frame);
        frame = roiDetector1.getForeground(false);
        cv::imshow("raw foreground",frame);
        std::cout<<count<<std::endl;
        count++;
    }


    return 0;
}