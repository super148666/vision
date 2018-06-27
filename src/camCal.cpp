//
// Created by chaoz on 6/12/17.
//

#include "camCal.h"

camCal::camCal() {
    this->chessboardSquareWidth = 0.0f;
    this->chessboardSize = cv::Size(0, 0);
}

camCal::camCal(float chessboardSquareWidth, cv::Size chessboardSize) {
    this->chessboardSquareWidth = chessboardSquareWidth;
    this->chessboardSize = chessboardSize;
}

camCal::~camCal() {

}

void camCal::getChessboardPosition(std::vector<cv::Point3f> &corners) {
    for (int i = 0; i < this->chessboardSize.height; i++) {
        for (int j = 0; j < this->chessboardSize.width; j++) {
            corners.push_back(cv::Point3f(i * this->chessboardSquareWidth, j * this->chessboardSquareWidth, 0.0f));
        }
    }
}

void camCal::getChessboardCorners(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> &allFoundCorners,
                                  bool showResults) {
    for (std::vector<cv::Mat>::iterator iter = images.begin(); iter != images.end(); iter++) {
        std::vector<cv::Point2f> pointBuf;
        bool found = findChessboardCorners(*iter, this->chessboardSize, pointBuf);

        if (found) {
            allFoundCorners.push_back(pointBuf);
        }


        if (showResults) {
            cv::Mat drawToImage = iter.base()->clone();
            drawChessboardCorners(drawToImage, this->chessboardSize, pointBuf, found);
            cv::imshow("press space to continue", drawToImage);
            cv::waitKey(0);
        }
    }
}

void camCal::cameraCalibration(std::vector<cv::Mat> calibrationImages) {
	std::cout<< 1;
    std::vector<std::vector<cv::Point2f>> localSpacePoints;
    this->getChessboardCorners(calibrationImages, localSpacePoints, false);
	std::cout<< 2;
    std::vector<std::vector<cv::Point3f>> worldSpacePoints(1);

    this->getChessboardPosition(worldSpacePoints[0]);
    worldSpacePoints.resize(localSpacePoints.size(), worldSpacePoints[0]);
	std::cout<< 3;
    std::vector<cv::Mat> rVectors, tVectors;
    distanceCoefficients = cv::Mat::zeros(8, 1, CV_64F);
	std::cout<< 4;
    calibrateCamera(worldSpacePoints, localSpacePoints, this->chessboardSize,
                    this->cameraMatrix, this->distanceCoefficients, rVectors,
                    tVectors);
	std::cout<< 5 <<std::endl;
}

bool camCal::saveCameraCalibration(std::string name) {
    std::ofstream outStream(name);
    if (outStream) {
        uint16_t rows = this->cameraMatrix.rows;
        uint16_t columns = this->cameraMatrix.cols;
        outStream << "{cameraMatrix:" << std::endl;
        for (int r = 0; r < rows; r++) {
            outStream << "  ";
            for (int c = 0; c < columns; c++) {
                double value = this->cameraMatrix.at<double>(r, c);
                outStream << value << " ";
            }
            outStream << std::endl;
        }
        outStream << "}" << std::endl;

        rows = this->distanceCoefficients.rows;
        columns = this->distanceCoefficients.cols;
        outStream << "{distanceCoefficients:" << std::endl;
        for (int r = 0; r < rows; r++) {
            outStream << "  ";
            for (int c = 0; c < columns; c++) {
                double value = this->distanceCoefficients.at<double>(r, c);
                outStream << value << " ";
            }
            outStream << std::endl;
        }
        outStream << "}" << std::endl;

        outStream.close();
        return true;
    }
    return false;
}
