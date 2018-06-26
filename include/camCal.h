//
// Created by chaoz on 6/12/17.
//

#ifndef CAM_CAL_CAMCAL_H
#define CAM_CAL_CAMCAL_H

#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

class camCal {
private:
    float chessboardSquareWidth;
    cv::Size chessboardSize;
    cv::Mat cameraMatrix;
    cv::Mat distanceCoefficients;
public:
    camCal();

    camCal(float chessboardSquareWidth, cv::Size chessboardSize);

    ~camCal();

    void getChessboardPosition(std::vector<cv::Point3f> &corners);

    void getChessboardCorners(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> &allFoundCorners,
                              bool showResults = false);

    void cameraCalibration(std::vector<cv::Mat> calibrationImages);
    
    cv::Mat getCameraMatrix() {
		return cameraMatrix.clone();
	}
	
	cv::Mat getDistanceCoefficients() {
		return distanceCoefficients.clone();
	}

    bool saveCameraCalibration(std::string name);

};


#endif //CAM_CAL_CAMCAL_H
