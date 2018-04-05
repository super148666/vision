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

bool WebcamEnable = false;
bool BackupEnable = false;
bool Sampling = false;

vector<Mat> detectedConesCells;
vector<string> emptyCones;
vector<string> emptyNonCones;
int numCones = 0;
int numNonCones = 0;
string pathCones("./cones/");
string pathNonCones("./non-cones/");

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
            Mat tempp = inputImg(windows[i]).clone();
            resize(tempp,tempp,Size(64,64));
            detectedConesCells.push_back(tempp);

        }
    }

}

void CollectSamples()
{
    if(detectedConesCells.empty()) return;
    int index = 0;
    vector<string> filename;
    int cellSize = detectedConesCells.size();
    filename.resize(cellSize,"not saved");
    namedWindow("samples",WINDOW_AUTOSIZE|WINDOW_NORMAL);
    char key = 0;
    while(1) {
        cout<<"current index: "<<index<<" | "<<filename[index]<<endl;
        cout<<"w:cone | s:non-cone | e:redo | space:next frame"<<endl;
        imshow("samples",detectedConesCells[index]);
        key = waitKey(0);
        switch(key){
            case 'a':
                if(index>0) index--;
                else index = cellSize-1;
                break;

            case 'd':
                if(index<(cellSize-1)) index++;
                else index = 0;
                break;

            case 'w':   //cones
                if(filename[index]=="not saved") {
                    if(emptyCones.empty())
                        filename[index] = pathCones + "image" + boost::lexical_cast<string>(++numCones) + ".png";
                    else {
                        filename[index]=emptyCones.front();
                        emptyCones.erase(emptyCones.begin());
                    }
                    imwrite(filename[index], detectedConesCells[index]);
                }
                else {
                    if(filename[index].find("non-")!=string::npos) {
                        emptyNonCones.push_back(filename[index]);
                        if(emptyCones.empty())
                            filename[index] = pathCones + "image" + boost::lexical_cast<string>(++numCones) + ".png";
                        else {
                            filename[index]=emptyCones.front();
                            emptyCones.erase(emptyCones.begin());
                        }
                        imwrite(filename[index], detectedConesCells[index]);
                    }
                }
                cout<<"image "<<index<<" has been saved at "<<filename[index]<<endl;
                if(index<(cellSize-1)) index++;
                else index = 0;
                break;

            case 's':   //non-cones
                if(filename[index]=="not saved") {
                    if(emptyNonCones.empty())
                        filename[index] = pathNonCones+"image" + boost::lexical_cast<string>(++numNonCones) + ".png";
                    else {
                        filename[index]=emptyNonCones.front();
                        emptyNonCones.erase(emptyNonCones.begin());
                    }
                    imwrite(filename[index], detectedConesCells[index]);
                }
                else {
                    if(filename[index].find("non-")==string::npos) {
                        emptyCones.push_back(filename[index]);
                        if(emptyNonCones.empty())
                            filename[index] = pathNonCones+"image" + boost::lexical_cast<string>(++numNonCones) + ".png";
                        else {
                            filename[index]=emptyNonCones.front();
                            emptyNonCones.erase(emptyNonCones.begin());
                        }
                        imwrite(filename[index], detectedConesCells[index]);
                    }
                }
                cout<<"image "<<index<<" has been saved at "<<filename[index]<<endl;
                if(index<(cellSize-1)) index++;
                else index = 0;
                break;

            case 'e':   //delete
                if(filename[index]=="not saved") {
                    cout<<"image "<<index<<" was not saved"<<endl;
                }
                else {
                    string command = "rm -rf "+filename[index];
                    system(command.c_str());
                    if(filename[index].find("non-")==string::npos)
                        emptyCones.push_back(filename[index]);
                    else
                        emptyNonCones.push_back(filename[index]);
                    filename[index] = "not saved";
                }
                break;

            case ' ':
                destroyWindow("samples");
                return;

            default:
                break;
        }
    }
}

int main(int argc, char** argv)
{
    bool enableSmall = false;
    bool enableMeduim = false;
    bool enableLarge = false;

    for(int i = 0; i < argc; i++)
    {
        if(string(argv[i])=="all")
        {
            enableSmall = true;
            enableMeduim = true;
            enableLarge = true;
        }

        if(string(argv[i])=="s")
        {
            enableSmall = true;
        }

        if(string(argv[i])=="m")
        {
            enableMeduim = true;
        }

        if(string(argv[i])=="l")
        {
            enableLarge = true;
        }
        
        if(string(argv[i])=="webcam")
        {
			WebcamEnable = true;
		}
		
		if(string(argv[i])=="backup")
		{
			BackupEnable = true;
		}

        if(string(argv[i])=="sampling")
        {
            Sampling = true;
            cout<<"number of cones:";
            cin>>numCones;
            cout<<"number of non-cones:";
            cin>>numNonCones;
        }

    }

    auto cap = WebcamEnable?VideoCapture(0):VideoCapture("./test.webm");
    Rect roi_large;
    Rect roi_medium;
    Rect roi_small;
    vector<Rect> windowsLarge;
    vector<Rect> windowsMedium;
    vector<Rect> windowsSmall;
    Ptr<SVM> svmLarge;
    Ptr<SVM> svmMedium;
    Ptr<SVM> svmSmall;
    if(BackupEnable) {
		roi_large=Rect(0,VideoHeight/20*15,VideoWidth,VideoHeight/20*5);
		roi_medium=Rect(0,VideoHeight/20*11,VideoWidth,VideoHeight/20*7);
		roi_small=Rect(0,VideoHeight/20*10,VideoWidth,VideoHeight/40*9);
		windowsSmall = get_sliding_windows(roi_small,SZ,SZ,1);
		windowsMedium = get_sliding_windows(roi_medium,MZ,MZ,3);
		windowsLarge = get_sliding_windows(roi_large,LZ,LZ,4);
		svmSmall = SVM::load("./backup/configure/model8.yml");
		svmMedium = SVM::load("./backup/configure/model12.yml");
		svmLarge = SVM::load("./backup/configure/model24.yml");
		hogSmall.load("./backup/configure/hogSmall.yml");
		hogMedium.load("./backup/configure/hogMedium.yml");
		hogLarge.load("./backup/configure/hogLarge.yml");
		cout<<cap.get(CV_CAP_PROP_POS_FRAMES);
		cout<<":"<<cap.get(CV_CAP_PROP_POS_MSEC)<<endl;
		cap.set(CV_CAP_PROP_POS_FRAMES,700);
		cout<<cap.get(CV_CAP_PROP_POS_FRAMES);
		cout<<":"<<cap.get(CV_CAP_PROP_POS_MSEC)<<endl;
	}
	else {
		svmSmall = SVM::load("./model8.yml");
		svmMedium = SVM::load("./model12.yml");
		svmLarge = SVM::load("./model24.yml");
		hogSmall.load("./hogSmall.yml");
		hogMedium.load("./hogMedium.yml");
		hogLarge.load("./hogLarge.yml");		
		roi_large=Rect(0,VideoHeight/20*8,VideoWidth,VideoHeight/20*12);
		roi_medium=Rect(0,VideoHeight/20*8,VideoWidth,VideoHeight/20*6);
		roi_small=Rect(0,VideoHeight/20*8,VideoWidth,VideoHeight/40*8);
		windowsSmall = get_sliding_windows(roi_small,SZ,SZ,1);
		windowsMedium = get_sliding_windows(roi_medium,MZ,MZ,4);
		windowsLarge = get_sliding_windows(roi_large,LZ,LZ,4);
	}
    const Mat emptyHeatMap(VideoHeight,VideoWidth,CV_8UC1,Scalar(0));
    int count = 0;
    while(cap.isOpened()) {
		if(BackupEnable) {
			if (count == 380) break;
		}
		Mat heatMap = emptyHeatMap.clone();
		Mat img,imgshow;
        cout<<"frame no:"<< ++count << endl;
        std::vector<cv::Point> Locations;
        bool success = cap.read(imgshow);
        if(!success) break;
        detectedConesCells.clear();
        resize(imgshow,imgshow,Size(VideoWidth,VideoHeight));
        imgshow.copyTo(img);
        Mat gray;
        cvtColor(img,gray,COLOR_BGR2GRAY);
        if(enableSmall)
        EvaluateSVM(img,imgshow,heatMap,windowsSmall,svmSmall,hogSmall);
        if(enableMeduim)
        EvaluateSVM(img,imgshow,heatMap,windowsMedium,svmMedium,hogMedium);
        if(enableLarge)
        EvaluateSVM(img,imgshow,heatMap,windowsLarge,svmLarge,hogLarge);

        for(int i=0;i<heatMap.rows;i++) {
            for(int j=0;j<heatMap.cols;j++) {
                if(heatMap.at<uint8_t>(i,j)>=2){
                    heatMap.at<uint8_t>(i,j) = gray.at<uint8_t>(i,j);
                }
                else {
                    heatMap.at<uint8_t>(i,j) = 0;
                }
            }
        }
        //if(enableLarge) rectangle(imgshow,roi_large,Scalar(0,255,0));
        //if(enableSmall) rectangle(imgshow,roi_small,Scalar(0,0,255));
        //if(enableMeduim) rectangle(imgshow,roi_medium,Scalar(255,0,0));

        //resize(imgshow,imgshow,Size(VideoWidth*4,VideoHeight*4));
        
        //resize(heatMap,heatMap,Size(VideoWidth*4,VideoHeight*4));
        imshow("img",imgshow);
        imshow("heat",heatMap);

        if(Sampling) CollectSamples();

		waitKey(1);
        imgshow.release();
		img.release();
    }
    return 0;

}
