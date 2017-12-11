#include <iostream>

#include "camCal.h"

using namespace std;
using namespace cv;

const float g_calibrationSquareWidth = 0.03508f;
const Size g_chessboardSize = Size(8, 6);

int main(int argv, char **argc) {
    camCal camCal1(g_calibrationSquareWidth,g_chessboardSize);
    int fps = 10;
    Mat frame;
    Mat drawToFrame;

    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distanceCoefficients;

    vector<Mat> savedImages;

    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    VideoCapture vid(0);

    if (!vid.isOpened()) {
        cout << "fail to open camera!" << endl;
        exit(0);
    }

    namedWindow("Webcam", WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL);

    while (true) {
        if (!vid.read(frame)) {
            break;
        }

        vector<Point2f> foundPoints;
        bool found = false;

        found = findChessboardCorners(frame, g_chessboardSize, foundPoints);
        frame.copyTo(drawToFrame);

        if (found) {
            drawChessboardCorners(drawToFrame, g_chessboardSize, foundPoints, found);
        }

        imshow("Webcam", drawToFrame);
        char cKey = waitKey(1000 / 60);

        switch (cKey) {
            case 13:    // enter
                //start calibration
                if (savedImages.size() < 50) {
                    cout << "More images required, try later" << endl;
                    break;
                }
                camCal1.cameraCalibration(savedImages);
                camCal1.saveCameraCalibration("camera_calibration_results");
                break;

            case 27:    // esc
                //exit
                return 0;
                break;

            case ' ':   // space
                //saving image
                if (found) {
                    Mat temp;
                    frame.copyTo(temp);
                    savedImages.push_back(temp);
                }
                cout<<" "<<savedImages.size()<<" have been collected!\n";
                break;

            default:
                break;
        }
    }

    return 0;
}