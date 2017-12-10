//
// Created by chaoz on 6/12/17.
//


#include "roiDetector.h"

roiDetector::roiDetector(int history, double varThreshold, bool detectShadows) {
    this->backgroundSubtractor = cv::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
    if (this->backgroundSubtractor == NULL) std::cout << "Fail to create bgSubtractor\n";
}

roiDetector::~roiDetector() {
}

int roiDetector::detect(cv::Mat *inputOutput, bool drawContours) {
    int count = 0;
    cv::cvtColor(*inputOutput,*inputOutput,CV_BGR2GRAY);
//    cv::medianBlur(*inputOutput,*inputOutput,5);
//    cv::cvtColor(*inputOutput, *inputOutput, CV_BGR2HLS);
//    cv::Mat colorImg[3];
//    cv::split(*inputOutput,colorImg);
//    *inputOutput = colorImg[0].clone();
    this->foreground = inputOutput->clone();
    this->backgroundSubtractor->apply(this->foreground, this->foreground, -1);
    this->backgroundSubtractor->getBackgroundImage(this->foreground);
//    cv::GaussianBlur(this->foreground,this->foreground,cv::Size(7,7),0.0,0.0);
    cv::Mat dx,dy;
    cv::Sobel(this->foreground,dx,CV_8U,1,0);
    cv::Sobel(this->foreground,dy,CV_8U,0,1);
    cv::addWeighted(dx,1.0,dy,0.2,0,this->foreground);
    cv::threshold(this->foreground,this->foreground,100,255,CV_THRESH_BINARY);
    this->roiMask = this->foreground.clone();
    this->close(this->roiMask,this->roiMask,1,cv::MORPH_RECT);
//    this->open(this->roiMask,this->roiMask,1,cv::MORPH_CROSS);
//    this->close(this->roiMask,this->roiMask,1,cv::MORPH_RECT);
//    this->close(this->roiMask,this->roiMask,1,cv::MORPH_RECT);
//    this->close(this->roiMask,this->roiMask,1,cv::MORPH_RECT);
    cv::findContours(this->roiMask, this->contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    if (drawContours) {
//        cv::drawContours(*inputOutput, this->contours, -1, cv::Scalar(0, 0, 255), -1);
    }
    for (std::vector<std::vector<cv::Point> >::iterator it = this->contours.begin(); it != this->contours.end(); it++) {
        count += (*it).size();
    }
    return count;
}

cv::Mat roiDetector::getForeground(bool raw) {
    if (raw) return this->foreground;
    return this->roiMask;
}

cv::Mat roiDetector::getBackground() {
    cv::Mat frame;
    this->backgroundSubtractor->getBackgroundImage(frame);
    return frame;
}

std::vector<std::vector<cv::Point> > roiDetector::getContours() {
    return this->contours;
}

int roiDetector::getSubtractHistroy() {
    return this->backgroundSubtractor->getHistory();
}

double roiDetector::getSubtractVarThreshold() {
    return this->backgroundSubtractor->getVarThreshold();
}

bool roiDetector::isDetectShadows() {
    return this->backgroundSubtractor->getDetectShadows();
}

void roiDetector::dilation(cv::Mat& input, cv::Mat& output, int size, int shape) {
    cv::dilate(input, output, cv::getStructuringElement(shape,
                                                        cv::Size(2 * size + 1, 2 * size + 1),
                                                        cv::Point(size, size)));
}

void roiDetector::erosion(cv::Mat& input, cv::Mat& output, int size, int shape) {
    cv::erode(input, output, cv::getStructuringElement(shape,
                                                       cv::Size(2 * size + 1, 2 * size + 1),
                                                       cv::Point(size, size)));
}

void roiDetector::open(cv::Mat& input, cv::Mat& output, int size, int shape) {
    cv::Mat element = cv::getStructuringElement(shape,
                                                cv::Size(2 * size + 1, 2 * size + 1),
                                                cv::Point(size, size));
    cv::erode(input, output, element);
    cv::dilate(output, output, element);
}

void roiDetector::close(cv::Mat& input, cv::Mat& output, int size, int shape) {
    cv::Mat element = cv::getStructuringElement(shape,
                                                cv::Size(2 * size + 1, 2 * size + 1),
                                                cv::Point(size, size));
    cv::dilate(input, output, element);
    cv::erode(output, output, element);
}


