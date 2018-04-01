//
// Created by chaoz on 28/03/18.
//

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>


#include <iostream>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <boost/lexical_cast.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

int SZ = 8;
int MZ = 12;
int LZ = 24;
int VideoWidth = 160;
int VideoHeight = 120;
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

Mat deskew(Mat& img){
    Moments m = moments(img);
    if(abs(m.mu02) < 1e-2){
        return img.clone();
    }
    float skew = m.mu11/m.mu02;
    Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);

    return imgOut;
}

HOGDescriptor hogSmall(
        Size(8,8), //winSize
        Size(4,4), //blocksize
        Size(2,2), //blockStride,
        Size(2,2), //cellSize,
        5, //nbins,
        1, //derivAper,
        -1, //winSigma,
        0, //histogramNormType,
        0.2, //L2HysThresh,
        0,//gammal correction,
        64,//nlevels=64
        1);

HOGDescriptor hogLarge(
        Size(8,8), //winSize
        Size(4,4), //blocksize
        Size(2,2), //blockStride,
        Size(2,2), //cellSize,
        5, //nbins,
        1, //derivAper,
        -1, //winSigma,
        0, //histogramNormType,
        0.2, //L2HysThresh,
        0,//gammal correction,
        64,//nlevels=64
        1);

HOGDescriptor hogMedium(
        Size(8,8), //winSize
        Size(4,4), //blocksize
        Size(2,2), //blockStride,
        Size(2,2), //cellSize,
        5, //nbins,
        1, //derivAper,
        -1, //winSigma,
        0, //histogramNormType,
        0.2, //L2HysThresh,
        0,//gammal correction,
        64,//nlevels=64
        1);

vector< float > get_svm_detector( const Ptr< SVM >& svm )
{
    Mat suppVecMat = svm->getSupportVectors();

    const float* ptrToFirstRow = suppVecMat.ptr<float>(0);
    vector<float> weightVector(ptrToFirstRow, ptrToFirstRow + suppVecMat.cols);
    vector<float> optionalOutputArrray;
    double rhoOfDecisionFunc = svm->getDecisionFunction(0, optionalOutputArrray, optionalOutputArrray);
// Rho of the decision function is the negative bias.
    double bias = -rhoOfDecisionFunc;
// We need to put the bias at the end of weightVector
    weightVector.push_back(bias);
    return weightVector;
}

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

void CreateSmallHOG(vector<vector<float> > &dataHOG, vector<Mat> &dataCells){

    dataHOG.clear();
    for(int y=0;y<dataCells.size();y++){
        vector<float> descriptors;
    	hogSmall.compute(dataCells[y],descriptors);
    	dataHOG.push_back(descriptors);
    }
}

void CreateLargeHOG(vector<vector<float> > &dataHOG, vector<Mat> &dataCells){

    dataHOG.clear();
    for(int y=0;y<dataCells.size();y++){
        vector<float> descriptors;
        hogLarge.compute(dataCells[y],descriptors);
        dataHOG.push_back(descriptors);
    }
}

void ConvertVectortoMatrix(vector<vector<float> > &dataHOG, Mat &dataMat)
{

    int descriptor_size = dataHOG[0].size();
    
    for(int i = 0;i<dataHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           dataMat.at<float>(i,j) = dataHOG[i][j]; 
        }
    }
}

void EvaluateSVM(Mat& inputImg, Mat& outputImg, Mat& heatMap, vector<Rect> windows, Ptr<SVM> svm, HOGDescriptor& hog) {
    vector<Mat> dataCells;
    Mat img;
    cvtColor(inputImg,img,COLOR_BGR2GRAY);

    for(size_t j=0;j<windows.size();j++) {
        Mat temp = img(windows[j]).clone();
        dataCells.push_back(temp);
    }

    vector<vector<float>> dataHOG;
    dataHOG.clear();
    for(int y=0;y<dataCells.size();y++){
        vector<float> descriptors;
        hog.compute(dataCells[y],descriptors);
        dataHOG.push_back(descriptors);
    }
    Mat dataMat(dataHOG.size(),dataHOG[0].size(),CV_32FC1);
    ConvertVectortoMatrix(dataHOG, dataMat);

    Mat response;
    svm->predict(dataMat,response);

    for(int i=0; i<windows.size();i++)
    {
        if(response.at<float>(i,0))
        {
            rectangle(outputImg,windows[i],Scalar(10,10,255));
            Mat temp(outputImg.rows,outputImg.cols,CV_8UC1,Scalar(0));
            rectangle(temp,windows[i],Scalar(1),-1);
            heatMap += temp;
        }
    }

}

int main(int argc, char** argv)
{
    auto cap = VideoCapture("./test.webm");
    Ptr<SVM> svmSmall = SVM::load("./model8.yml");
    Ptr<SVM> svmMedium = SVM::load("./model12.yml");
    Ptr<SVM> svmLarge = SVM::load("./model24.yml");
    const Rect roi_large(0,VideoHeight/20*10,VideoWidth,VideoHeight/20*10);
    const Rect roi_medium(0,VideoHeight/20*9,VideoWidth,VideoHeight/20*8);
    const Rect roi_small(0,VideoHeight/20*9,VideoWidth,VideoHeight/40*7);
    vector<Rect> windowsSmall = get_sliding_windows(roi_small,SZ,SZ,1);
    vector<Rect> windowsMedium = get_sliding_windows(roi_medium,MZ,MZ,3);
    vector<Rect> windowsLarge = get_sliding_windows(roi_large,LZ,LZ,4);
    hogSmall.load("./hogSmall.yml");
    hogMedium.load("./hogMedium.yml");
    hogLarge.load("./hogLarge.yml");
    const Mat emptyHeatMap(VideoHeight,VideoWidth,CV_8UC1,Scalar(0));
    int count = 0;
    while(cap.isOpened()) {
		Mat heatMap = emptyHeatMap.clone();
		Mat img,imgshow;
        cout<<"frame no:"<< ++count << endl;
        std::vector<cv::Point> Locations;
        bool success = cap.read(imgshow);
        if(!success) break;

        resize(imgshow,imgshow,Size(VideoWidth,VideoHeight));
        imgshow.copyTo(img);

        EvaluateSVM(img,imgshow,heatMap,windowsSmall,svmSmall,hogSmall);
        EvaluateSVM(img,imgshow,heatMap,windowsMedium,svmMedium,hogMedium);
//        EvaluateSVM(img,imgshow,heatMap,windowsLarge,svmLarge,hogLarge);

        for(int i=0;i<heatMap.rows;i++) {
            for(int j=0;j<heatMap.cols;j++) {
                if(heatMap.at<uint8_t>(i,j)>=2){
                    heatMap.at<uint8_t>(i,j) = img.at<uint8_t>(i,j);
                }
                else {
                    heatMap.at<uint8_t>(i,j) = 0;
                }
            }
        }
//        rectangle(imgshow,roi_large,Scalar(0,255,0));
//        rectangle(imgshow,roi_small,Scalar(0,0,255));
//        rectangle(imgshow,roi_medium,Scalar(255,0,0));
        imshow("heat",imgshow);
		waitKey(1);
		imgshow.release();
		img.release();
    }
    return 0;

}
