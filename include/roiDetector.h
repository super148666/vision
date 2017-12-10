//
// Created by chaoz on 6/12/17.
//

#ifndef CAM_CAL_ROIDETECTOR_H
#define CAM_CAL_ROIDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <iostream>


class roiDetector {
private:
    cv::Ptr<cv::BackgroundSubtractorMOG2> backgroundSubtractor;
    cv::Mat foreground;
    cv::Mat roiMask;
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat erosionElement;
    cv::Mat dilationElement;
public:

    roiDetector(int history, double varThreshold,
                bool detectShadows);

    ~roiDetector();

    int detect(cv::Mat *inputOutput, bool drawContours);

    cv::Mat getForeground(bool raw = true);

    cv::Mat getBackground();

    std::vector<std::vector<cv::Point> > getContours();

    int getSubtractHistroy();

    double getSubtractVarThreshold();

    bool isDetectShadows();

    void dilation(cv::Mat& input, cv::Mat& output, int size, int shape);

    void erosion(cv::Mat& input, cv::Mat& output, int size, int shape);

    void open(cv::Mat& input, cv::Mat& output, int size, int shape);

    void close(cv::Mat& input, cv::Mat& output, int size, int shape);
};


#endif //CAM_CAL_ROIDETECTOR_H
