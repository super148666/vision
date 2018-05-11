//
// Created by chaoz on 28/03/18.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

#include <iostream>
#include <string>
#include <boost/lexical_cast.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;
using namespace visualization_msgs;

/*
#define COLOR_THRESHOLD 10
#define PROCESS_SIZE 64
#define SCALE_X 0.6
#define SCALE_Y 0.6
#define WINDOW_SIZE 64
#define TARGET_H 42
#define TARGET_V 251
*/


int color_threshold = 1;
int process_size = 64;
int window_stride = 16;
double scale_x = 1.0;
double scale_y = 1.0;
double roi_upper = 0.5;
double roi_lower = 1.0;
int target_hue = 42;
int target_value = 250;
bool debug_on = true;
bool webcam_on = false;
int webcam_port = 1;
std::string video_filename = "test.webm";
std::string hog_filename = "hog64.yaml";
std::string svm_filename = "model64.yaml";

int UpperMargin = 0;
int LowerMargin = 0;
int LeftMargin = 0;
int RightMargin = 0;
int VideoHeight = 0;
int VideoWidth = 0;
int ProcessHeight = 0;
int ProcessWidth = 0;
vector<Point2f> pts_src;
vector<Point2f> pts_dst;

VideoCapture cap;
Mat img, h;
vector<Mat> detectedConesCells;
Rect roi;
vector<Rect> windows;
vector<Point2f> windowsStartPoints;
Ptr<SVM> svm;

ros::Publisher cones_pub;
MarkerArray cones_array;
Marker marker_template;

struct passwd *pw = getpwuid(getuid());

const char *homedir = pw->pw_dir;

HOGDescriptor hog(
        Size(8, 8), //winSize
        Size(4, 4), //blocksize
        Size(2, 2), //blockStride,
        Size(2, 2), //cellSize,
        5, //nbins,
        1, //derivAper,
        -1, //winSigma,
        0, //histogramNormType,
        0.2, //L2HysThresh,
        0,//gammal correction,
        64,//nlevels=64
        1);

vector<float> get_svm_detector(const Ptr<SVM> &svm) {
    Mat suppVecMat = svm->getSupportVectors();

    const float *ptrToFirstRow = suppVecMat.ptr<float>(0);
    vector<float> weightVector(ptrToFirstRow, ptrToFirstRow + suppVecMat.cols);
    vector<float> optionalOutputArrray;
    double rhoOfDecisionFunc = svm->getDecisionFunction(0, optionalOutputArrray, optionalOutputArrray);
// Rho of the decision function is the negative bias.
    double bias = -rhoOfDecisionFunc;
// We need to put the bias at the end of weightVector
    weightVector.push_back(bias);
    return weightVector;
}

void imshow_debug(const char* str, Mat input) {
    if(debug_on)
        imshow(str,input);
}

std::vector<cv::Rect>
get_sliding_windows(Rect roi, vector<Point2f> &winStartPoints, int winWidth, int winHeight, int step) {
    std::vector<cv::Rect> rects;
    if (!winStartPoints.empty()) winStartPoints.clear();
    for (int i = roi.y; i < (roi.y + roi.height); i += step) {
        if ((i + winHeight) > (roi.y + roi.height)) { break; }
        for (int j = roi.x; j < (roi.x + roi.width); j += step) {
            if ((j + winWidth) > (roi.x + roi.width)) { break; }
            cv::Rect rect(j, i, winWidth, winHeight);
            winStartPoints.push_back(Point2f(j, i));
            rects.push_back(rect);
        }
    }
    return rects;
}

void ConvertVectortoMatrix(vector<vector<float> > &dataHOG, Mat &dataMat) {

    int descriptor_size = dataHOG[0].size();

    for (int i = 0; i < dataHOG.size(); i++) {
        for (int j = 0; j < descriptor_size; j++) {
            dataMat.at<float>(i, j) = dataHOG[i][j];
        }
    }
}

Point2f GetColorCenter(Mat input, uint8_t target_h, uint8_t target_v, uint8_t threshold) {
    //cvtColor(input, input, COLOR_BGR2HSV_FULL);
    vector<int> hist_rows;
    vector<int> hist_cols;
    vector<Mat> spl;
    split(input, spl);
    hist_rows.clear();
    hist_cols.clear();
    hist_rows.resize(input.rows, 0);
    hist_cols.resize(input.cols, 0);
    bool match = false;
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {

            double b_ = spl[0].at<uint8_t>(i,j);
            double g_ = spl[1].at<uint8_t>(i,j);
            double r_ = spl[2].at<uint8_t>(i,j);
            double c_max = max(r_,max(g_,b_));
            double c_min = min(r_,max(g_,b_));
            double delta = c_max-c_min;
            uint8_t hue = 0;
            uint8_t sat = (1.0 - 3 * c_min / (b_ + g_ + r_)) * 255;
            uint8_t intensity = (b_ + g_ + r_) / 3.0;
            if(6*delta<=c_max) hue = 255;
            else{
                if(r_==c_max) hue = 42 + 42 * ((g_-b_) / delta);
                else if(g_==c_max) hue = 126 + 42 * ((b_-r_) / delta);
                else if(b_==c_max) hue = 210 + 42 * ((r_-g_) / delta);
            }

            spl[0].at<uint8_t>(i, j) = hue;
            spl[1].at<uint8_t>(i, j) = sat;
            spl[2].at<uint8_t>(i, j) = intensity;

            int diff1 = spl[0].at<uint8_t>(i, j) - target_h;
            int diff2 = spl[2].at<uint8_t>(i, j);

            if (abs(diff1) < threshold) {
                if (diff2 > target_v) {
                    hist_rows[i]++;
                    hist_cols[j]++;
                    if (!match) match = true;
                }
            }
        }
    }




    if (match) {
        int max = 0;
        int max_length = 0;
        int length = 0;
        int index_y = 0;
        for (int i = 0; i < hist_rows.size(); i++) {
            if (hist_rows[i] >= max) {
                index_y = i;
                max = hist_rows[i];
                max_length = 1;
                length = 1;
            //} else if (hist_rows[i] == max) {
            //    length++;
            //} else {
            //    if (length > max_length)
            //        max_length = length;
            }
        }
        index_y += max_length / 2;

        max = 0;
        max_length = 0;
        length = 0;
        int index_x = 0;
        for (int i = 0; i < hist_cols.size(); i++) {
            if (hist_cols[i] > max) {
                index_x = i;
                max = hist_cols[i];
                max_length = 1;
                length = 1;
//            } else if (hist_cols[i] == max) {
//                length++;
//            } else {
//                if (length > max_length) {
//                    max_length = length;
//                }
            }
        }
        index_x += max_length / 2;
        imshow_debug("hue",spl[0]);
        imshow_debug("sat",spl[1]);
        imshow_debug("int",spl[2]);
        return Point2f(index_x, index_y);
    } else {
        return Point2f(-1, -1);
    }
}


void EvaluateSVM(Mat &inputImg, Mat &outputImg, Mat &heatMap, vector<Rect> windows, vector<Point2f> winStartPoints,
                 vector<Point2f> &conePoints, Ptr<SVM> svm, HOGDescriptor hog) {
    vector<Mat> dataCells;
    Mat img;
    cvtColor(inputImg, img, COLOR_BGR2GRAY);
    //ROS_INFO("convert img to gray");

    for (size_t j = 0; j < windows.size(); j++) {
        Mat temp = img(windows[j]).clone();
        dataCells.push_back(temp);
    }
    //ROS_INFO_STREAM("get img patch"<<dataCells.size());

    vector<vector<float>> dataHOG;
    dataHOG.clear();
    for (int y = 0; y < dataCells.size(); y++) {
        vector<float> descriptors;
        hog.compute(dataCells[y], descriptors);
        dataHOG.push_back(descriptors);
    }
    //ROS_INFO_STREAM("compute hog vector"<<dataHOG.size()<<" "<<dataHOG[0].size());

    Mat dataMat(dataHOG.size(), dataHOG[0].size(), CV_32FC1);
    ConvertVectortoMatrix(dataHOG, dataMat);
    //ROS_INFO("convert hog vector into matrix");

    Mat response;
    svm->predict(dataMat, response);
    //ROS_INFO("svm predict");

    conePoints.clear();
    for (int i = 0; i < windows.size(); i++) {
        if (response.at<float>(i, 0)) {

            Mat temp(outputImg.rows, outputImg.cols, CV_8UC1, Scalar(0));

            Mat tempp = inputImg(windows[i]).clone();
            //resize(tempp,tempp,Size(64,64));

            Point2f center = GetColorCenter(tempp, target_hue, target_value, color_threshold);
            if (center.x >= 0) {
                detectedConesCells.push_back(tempp);
                /*
                cvtColor(tempp,tempp,COLOR_BGR2HSV_FULL);
                vector<Mat> spl;
                split(tempp,spl);
                imshow("h",spl[0]);
                imshow("s",spl[1]);
                imshow("v",spl[2]);
                */
                //ROS_INFO_STREAM("x:"<<center.x<<" y:"<<center.y);
                center.x += winStartPoints[i].x;
                center.y += winStartPoints[i].y;
                rectangle(temp, windows[i], Scalar(1), -1);
                heatMap += temp;
                rectangle(outputImg, windows[i], Scalar(10, 10, 255));
                circle(outputImg, center, 2, Scalar(10, 10, 255));
                conePoints.push_back(center);
            }
        }
    }
    //ROS_INFO("mark positive prediction");

}

void CalibrateLocation(vector<Point2f> input, vector<Point2f> &output) {
    if (input.empty()) {
        ROS_INFO("empty input for CalibrateLocation function");
        return;
    }
    if (!output.empty()) {
        output.clear();
    }
    static const double y_scale = 1.6 / (double) (RightMargin - LeftMargin);
    static const double x_scale = 0.50 / (double) (LowerMargin - UpperMargin);
    for (int i = 0; i < input.size(); i++) {
        Point2f temp;
        temp.y = -(input[i].x - (LeftMargin + RightMargin) / 2);
        temp.x = -(input[i].y - LowerMargin);
        //ROS_INFO_STREAM("in img x:"<<temp.y<<" y:"<<temp.x);
        temp.y = temp.y * y_scale;
        temp.x = temp.x * x_scale + 5.0;
        output.push_back(temp);
        //ROS_INFO_STREAM("to car x:"<<temp.x<<" y:"<<temp.y);
    }
}

void EvaluateConeLocation(vector<Point2f> coneInImg, vector<Point2f> &coneInBaselink, Mat transform) {
    if (coneInImg.empty()) {
        ROS_INFO("no available cones");
        return;
    }
    vector<Point2f> coneInTransform;
    coneInTransform.clear();
    perspectiveTransform(coneInImg, coneInTransform, transform);
    CalibrateLocation(coneInTransform, coneInBaselink);

}

void PublishCone(vector<Point2f> coneInBaselink) {
    if (!cones_array.markers.empty()) {
        for (int i = 0; i < cones_array.markers.size(); i++) {
            cones_array.markers[i].action = Marker::DELETE;
        }
        cones_pub.publish(cones_array);
        cones_array.markers.clear();
    }

    if (coneInBaselink.empty()) {
        ROS_INFO("no cone to publish");
        return;
    }

    for (int i = 0; i < coneInBaselink.size(); i++) {
        Marker temp_marker = marker_template;
        temp_marker.ns = "raw_cones";
        temp_marker.id = i;
        temp_marker.pose.position.x = coneInBaselink[i].x;
        temp_marker.pose.position.y = coneInBaselink[i].y;
        double confidence;
        if(coneInBaselink[i].x < 3.0) confidence = 1.0;
        else if (coneInBaselink[i].x > 10.0) confidence = 0.1;
        else confidence = (10.0 - coneInBaselink[i].x) * 0.9 / 7.0 + 0.1;
        temp_marker.color.r = temp_marker.color.r * confidence;
        temp_marker.color.b = temp_marker.color.b * confidence;
        cones_array.markers.push_back(temp_marker);
    }
    cones_pub.publish(cones_array);
}

void initGlobalVariables(ros::NodeHandle nh) {

    std::string pkg_path = ros::package::getPath("vision")+"/";

    nh.getParam("color_threshold",color_threshold);
    nh.getParam("process_size",process_size);
    nh.getParam("window_stride",window_stride);
    nh.getParam("roi_upper",roi_upper);
    nh.getParam("roi_lower",roi_lower);
    roi_lower = roi_lower - roi_upper;
    if(roi_lower<=0||roi_upper<0) {
        ROS_ERROR("wrong roi configure");
        exit(-1);
    }
    nh.getParam("scale_x",scale_x);
    nh.getParam("scale_y",scale_y);
    nh.getParam("target_hue",target_hue);
    nh.getParam("target_value",target_value);
    nh.getParam("debug",debug_on);
    nh.getParam("webcam_on",webcam_on);
    nh.getParam("webcam_port",webcam_port);
    nh.getParam("video_filename",video_filename);
    nh.getParam("hog_filename",hog_filename);
    nh.getParam("svm_filename",svm_filename);
    video_filename.insert(0,pkg_path);
    hog_filename.insert(0,pkg_path);
    svm_filename.insert(0,pkg_path);

    if(webcam_on)
        cap = VideoCapture(webcam_port);
    else
        cap = VideoCapture(video_filename);
    if (!cap.read(img)) exit(1);

    VideoWidth = img.cols;
    VideoHeight = img.rows;
    ProcessHeight = VideoHeight * scale_y;
    ProcessWidth = VideoWidth * scale_x;

    LeftMargin = (VideoWidth * 4 / 10) * scale_x;
    RightMargin = (VideoWidth - VideoWidth * 4 / 10) * scale_x;
    UpperMargin = (VideoHeight * 4 / 10) * scale_y;
    LowerMargin = (VideoHeight - VideoHeight * 4 / 10) * scale_y;
    pts_src.clear();
    pts_src.push_back(Point2f(LeftMargin, UpperMargin));
    pts_src.push_back(Point2f(RightMargin, UpperMargin));
    pts_src.push_back(Point2f(RightMargin, LowerMargin));
    pts_src.push_back(Point2f(LeftMargin, LowerMargin));

    pts_dst.clear();
    pts_dst.push_back(Point2f((int) (235 * scale_x), (int) (320 * scale_y)));
    pts_dst.push_back(Point2f((int) (435 * scale_x), (int) (322 * scale_y)));
    pts_dst.push_back(Point2f((int) (448 * scale_x), (int) (345 * scale_y)));
    pts_dst.push_back(Point2f((int) (225 * scale_x), (int) (342 * scale_y)));

    h = getPerspectiveTransform(pts_dst, pts_src);


    /*
        int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        VideoWriter video(o1,CV_FOURCC('M','J','P','G'),30, Size(frame_width,frame_height),true);
    */
    if(debug_on) {
        namedWindow("video", WINDOW_AUTOSIZE);
        namedWindow("transformed", WINDOW_AUTOSIZE);
    }

    hog.load(hog_filename);
    roi = Rect(0, ProcessHeight * roi_upper, ProcessWidth, ProcessHeight * roi_lower);
    windows = get_sliding_windows(roi, windowsStartPoints, process_size, process_size, window_stride);
    svm = SVM::load(svm_filename);

    marker_template.action = visualization_msgs::Marker::ADD;
    marker_template.header.frame_id = "base_link";
    marker_template.type = visualization_msgs::Marker::CYLINDER;
    tf::poseTFToMsg(tf::Transform(tf::createQuaternionFromRPY(0.0, 0.0, 0.0),
                                  tf::Vector3(0.0, 0.0, 0.0)),
                    marker_template.pose);
    marker_template.scale.x = 0.2;
    marker_template.scale.y = 0.2;
    marker_template.scale.z = 0.5;
    marker_template.color.a = 1.0;
    marker_template.color.b = 1.0;
    marker_template.color.r = 1.0;
    marker_template.color.g = 0.0;
}



int main(int argc, char **argv) {
    ros::init(argc, argv, "working");
    ros::NodeHandle nh;
    ros::Rate rate_(30);
    cones_pub = nh.advertise<MarkerArray>("raw_cones", 10);


    initGlobalVariables(nh);

    ROS_INFO("init done");

    char key = 0;

    const Mat emptyHeatMap(ProcessHeight, ProcessWidth, CV_8UC1, Scalar(0));
    while (key != ' ' && cap.isOpened()) {

        if (!cap.read(img)) exit(1);
        //ROS_INFO_STREAM("frame no."<<count++);

        std::vector<cv::Point> Locations;

        detectedConesCells.clear();
        Mat img_, img_process, img_show;
        img_ = img.clone();
        resize(img_, img_process, Size(ProcessWidth, ProcessHeight));
        img_show = img_process.clone();
        Mat heatMap = emptyHeatMap.clone();
        //ROS_INFO("preprocessing done");
        vector<Point2f> conePoints;
        EvaluateSVM(img_process, img_show, heatMap, windows, windowsStartPoints, conePoints, svm, hog);
        //ROS_INFO("SVM evaluation done");
        vector<Point2f> coneToSAE;
        EvaluateConeLocation(conePoints, coneToSAE, h);
        PublishCone(coneToSAE);
//        Mat out;

        //warpPerspective(img_show, out, h, img.size());
        //ROS_INFO("perspective transform done");

        //imshow("transformed",out);
        //moveWindow("transformed",100,100);
        imshow_debug("video", img_show);


        /*
        resize(img, img, Size(VideoWidth/2, VideoHeight/2));
        vector<Mat> spl;
        vector<Mat> spl2;

        split(img,spl);
        spl2.resize(3);
        equalizeHist(spl[0],spl2[0]);
        equalizeHist(spl[1],spl2[1]);
        equalizeHist(spl[2],spl2[2]);
        merge(spl2,img);
        cvtColor(img,img,COLOR_BGR2Luv);
        split(img,spl);
        imshow("0",spl[0]);
        imshow("1",spl[1]);
        imshow("2",spl[2]);
        */

        key = waitKey(1);
        //rate_.sleep();
    }
    return 0;

}

