
//This code will run in both opencv 2 and 3. Just change the first two macros in the code according to the requirement.

#define USE_OPENCV_3
//#define USE_OPENCV_2


#ifdef USE_OPENCV_3
   #include <iostream>
   #include <opencv2/highgui.hpp>
   #include <opencv2/imgproc.hpp>
   #include "opencv2/objdetect.hpp"
   #include <opencv2/ml.hpp>
#endif

#ifdef USE_OPENCV_2
   #include <cv.h>
   #include <highgui.h>
   #include <opencv2/ml/ml.hpp>
#endif

#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <random>       // std::default_random_engine
#include <thread>
#include <chrono>       // std::chrono::system_clock
#include <boost/lexical_cast.hpp>

#ifdef USE_OPENCV_3
using namespace cv::ml;
#endif
using namespace cv;
using namespace std;

using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;


string hog_filename = "param/hog.yml";
string svm_filename = "param/svm.yml";

int num_of_dataset = 0;
vector<string> pathNameCones;
vector<int> numCones;
vector<string> pathNameNonCones;
vector<int> numNonCones;
int SZ = 64;
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;
float trainSetSize = 0.9;
float Gamma = 1.0;
float Ccoef = 5.0;

Mat deskew(Mat& img){

//    Moments m = moments(img);
//    if(abs(m.mu02) < 1e-2){
//        return img.clone();
//    }
//    float skew = m.mu11/m.mu02;
//    Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
//    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
//    warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);
//
//    return imgOut;

    medianBlur(img,img,5);
    return img;
}

void loadDataLabel(vector<string> &pathNameCones, vector<int> numCones, vector<string> &pathNameNonCones, vector<int> numNonCones, vector<Mat> &trainCells, vector<Mat> &validateCells, vector<Mat> &testCells, vector<int> &trainLabels, vector<int> &validateLabels, vector<int> &testLabels){
	/* Cones */
	vector<Mat> allCones;
    for(int j = 0; j < num_of_dataset; j++) {
        for (int i = 0; i < numCones[j]; i++) {
            string imagePath;
            //if (i < 10)
            //    imagePath = pathNameCones[j] + "image00" + boost::lexical_cast<string>(i) + ".png";
            //else if (i < 100)
            //    imagePath = pathNameCones[j] + "image0" + boost::lexical_cast<string>(i) + ".png";
            //else
            imagePath = pathNameCones[j] + "image" + boost::lexical_cast<string>(i) + ".png";
            Mat img = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
            if (!img.empty()) {
                resize(img, img, Size(SZ, SZ));
                allCones.push_back(img);
            }
        }
    }
	// shuffle
	// 80% into train set; 10% into validate set; 10% into test set
	random_shuffle(allCones.begin(), allCones.end());
	const size_t trainSizeCones = allCones.size() * trainSetSize;
	const size_t validateSizeCones = allCones.size() * 0;
	const size_t testSizeCones = allCones.size() - trainSizeCones - validateSizeCones;
	//clear
	trainCells.clear();
	validateCells.clear();
	testCells.clear();
	//reserve
	trainCells.reserve(trainSizeCones);
	validateCells.reserve(validateSizeCones);
	testCells.reserve(testSizeCones);
	//insert
	trainCells.insert(trainCells.end(), allCones.begin(), allCones.begin() + trainSizeCones);
	validateCells.insert(validateCells.end(), allCones.begin() + trainSizeCones, allCones.begin() + trainSizeCones + validateSizeCones);
	testCells.insert(testCells.end(), allCones.begin() + trainSizeCones + validateSizeCones, allCones.end());
	// setup labels -- cones == 1
	trainLabels.clear();
	trainLabels.resize(trainSizeCones, 1);
	validateLabels.clear();
	validateLabels.resize(validateSizeCones, 1);
	testLabels.clear();
	testLabels.resize(testSizeCones, 1);
	/* End Cones */
	
	/* NonCones */
	vector<Mat> allNonCones;
    for(int j = 0; j < num_of_dataset; j++) {
        for (int i = 0; i < numNonCones[j]; i++) {
            string imagePath;
            imagePath = pathNameNonCones[j] + "image" + boost::lexical_cast<string>(i) + ".png";
            Mat img = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
            if (!img.empty()) {
                resize(img, img, Size(SZ, SZ));
                allNonCones.push_back(img);
            }
        }
    }
	// shuffle
	// 80% into train set; 10% into validate set; 10% into test set
	random_shuffle(allNonCones.begin(), allNonCones.end());
	const size_t trainSizeNonCones = allNonCones.size() * trainSetSize;
	const size_t validateSizeNonCones = allNonCones.size() * 0;
	const size_t testSizeNonCones = allNonCones.size() - trainSizeNonCones - validateSizeNonCones;
	//reserve
	trainCells.reserve(trainSizeCones+trainSizeNonCones);
	validateCells.reserve(validateSizeCones+validateSizeNonCones);
	testCells.reserve(testSizeCones+testSizeNonCones);
	//insert
	trainCells.insert(trainCells.end(), allNonCones.begin(), allNonCones.begin() + trainSizeNonCones);
	validateCells.insert(validateCells.end(), allNonCones.begin() + trainSizeNonCones, allNonCones.begin() + trainSizeNonCones + validateSizeNonCones);
	testCells.insert(testCells.end(), allNonCones.begin() + trainSizeNonCones + validateSizeNonCones, allNonCones.end());
	// setup labels -- cones == 1
	trainLabels.resize(trainSizeCones+trainSizeNonCones, 0);
	validateLabels.resize(validateSizeCones+validateSizeNonCones, 0);
	testLabels.resize(testSizeCones+testSizeNonCones, 0);
	/* End NonCones */
	
	/* shuffle train set */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(trainCells.begin(), trainCells.end(), std::default_random_engine(seed));
	shuffle(trainLabels.begin(), trainLabels.end(), std::default_random_engine(seed));
	/* End shuffle train set */
	
	/* shuffle validate set */
	seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(validateCells.begin(), validateCells.end(), std::default_random_engine(seed));
	shuffle(validateLabels.begin(), validateLabels.end(), std::default_random_engine(seed));
	/* End shuffle validate set */

	/* shuffle test set */
	seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(testCells.begin(), testCells.end(), std::default_random_engine(seed));
	shuffle(testLabels.begin(), testLabels.end(), std::default_random_engine(seed));
	/* End shuffle test set */
}

void CreateDeskewedData(vector<Mat> &deskewedTrainCells, vector<Mat> &deskewedValidateCells, vector<Mat> &deskewedTestCells, vector<Mat> &trainCells, vector<Mat> &validateCells, vector<Mat> &testCells){
    

    for(int i=0;i<trainCells.size();i++){

    	Mat deskewedImg = deskew(trainCells[i]);
    	deskewedTrainCells.push_back(deskewedImg);
    }
    
    for(int i=0;i<validateCells.size();i++){

    	Mat deskewedImg = deskew(validateCells[i]);
    	deskewedValidateCells.push_back(deskewedImg);
    }

    for(int i=0;i<testCells.size();i++){

    	Mat deskewedImg = deskew(testCells[i]);
    	deskewedTestCells.push_back(deskewedImg);
    }
}

HOGDescriptor hog(
        Size(SZ,SZ), //winSize
        Size(32,32), //blocksize
        Size(16,16), //blockStride,
        Size(16,16), //cellSize,
                 10, //nbins,
                  1, //derivAper,
                 -1, //winSigma,
                  0, //histogramNormType,
                0.2, //L2HysThresh,
                  0,//gammal correction,
                  64,//nlevels=64
                  1);
                  
void CreateDataHOG(vector<vector<float> > &trainHOG, vector<vector<float> > &validateHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedTrainCells, vector<Mat> &deskewedValidateCells, vector<Mat> &deskewedTestCells){

    trainHOG.clear();
    for(int y=0;y<deskewedTrainCells.size();y++){
        vector<float> descriptors;
    	hog.compute(deskewedTrainCells[y],descriptors);
    	trainHOG.push_back(descriptors);
    }
    
    validateHOG.clear();
    for(int y=0;y<deskewedValidateCells.size();y++){
        vector<float> descriptors;
    	hog.compute(deskewedValidateCells[y],descriptors);
    	validateHOG.push_back(descriptors);
    }
    
    testHOG.clear();
    for(int y=0;y<deskewedTestCells.size();y++){
    	
        vector<float> descriptors;
    	hog.compute(deskewedTestCells[y],descriptors);
    	testHOG.push_back(descriptors);
    } 
}

void ConvertVectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &validateHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &validateMat, Mat &testMat)
{

    int descriptor_size = trainHOG[0].size();
    
    for(int i = 0;i<trainHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           trainMat.at<float>(i,j) = trainHOG[i][j]; 
        }
    }
    
    for(int i = 0;i<validateHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           validateMat.at<float>(i,j) = validateHOG[i][j]; 
        }
    }
    
    for(int i = 0;i<testHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
            testMat.at<float>(i,j) = testHOG[i][j]; 
        }
    }
}

void getSVMParams(SVM *svm)
{
    cout << "Kernel type     : " << svm->getKernelType() << endl;
    cout << "Type            : " << svm->getType() << endl;
    cout << "C               : " << svm->getC() << endl;
    cout << "Degree          : " << svm->getDegree() << endl;
    cout << "Nu              : " << svm->getNu() << endl;
    cout << "Gamma           : " << svm->getGamma() << endl;
}

void SVMtrain(Mat &trainMat,vector<int> &trainLabels, Mat &testResponse,Mat &testMat){
#ifdef USE_OPENCV_2
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.gamma = 0.50625;
    params.C = 2.5;
    CvSVM svm;
    CvMat tryMat = trainMat;
    Mat trainLabelsMat(trainLabels.size(),1,CV_32FC1);

    for(int i = 0; i< trainLabels.size();i++){
        trainLabelsMat.at<float>(i,0) = trainLabels[i];
    }
    CvMat tryMat_2 = trainLabelsMat;
    svm.train(&tryMat,&tryMat_2, Mat(), Mat(), params);
    svm.predict(testMat,testResponse);
#endif
#ifdef USE_OPENCV_3
    Ptr<SVM> svm = SVM::create();
    svm->setGamma(Gamma); //0.50625
    svm->setC(Ccoef);
    svm->setKernel(SVM::LINEAR);
    svm->setType(SVM::C_SVC);
    Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
    svm->train(td);
    svm->save(svm_filename);
    hog.save(hog_filename);
    svm->predict(testMat, testResponse);
    //getSVMParams(svm);
#endif
}


void SVMevaluate(Mat &testResponse,float &count, float &accuracy,vector<int> &testLabels){

    for(int i=0;i<testResponse.rows;i++)
    {
        //cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
        if(testResponse.at<float>(i,0) == testLabels[i]){
            count = count + 1;
        }  
    }
    accuracy = (count/testResponse.rows)*100;
}

void InitGlobalVariables(ros::NodeHandle nh) {
    std::string pkg_path = ros::package::getPath("vision")+"/";

    nh.getParam("num_of_datasets",num_of_dataset);
    numNonCones.reserve(num_of_dataset);
    numCones.reserve(num_of_dataset);
    pathNameNonCones.reserve(num_of_dataset);
    pathNameCones.reserve(num_of_dataset);
    string str_temp;
    int int_temp=0;
    nh.getParam("/first/num_of_cones",int_temp);
    numCones.push_back(int_temp);
    int_temp = 0;
    nh.getParam("/first/num_of_noncones",int_temp);
    numNonCones.push_back(int_temp);
    nh.getParam("/first/path_of_cones",str_temp);
    pathNameCones.push_back(str_temp);
    str_temp.clear();
    nh.getParam("/first/path_of_noncones",str_temp);
    pathNameNonCones.push_back(str_temp);
    if(num_of_dataset==2) {
        int_temp=0;
        nh.getParam("/second/num_of_cones",int_temp);
        numCones.push_back(int_temp);
        int_temp = 0;
        nh.getParam("/second/num_of_noncones",int_temp);
        numNonCones.push_back(int_temp);
        nh.getParam("/second/path_of_cones",str_temp);
        pathNameCones.push_back(str_temp);
        str_temp.clear();
        nh.getParam("/second/path_of_noncones",str_temp);
        pathNameNonCones.push_back(str_temp);
    }
    nh.getParam("sample_size",SZ);
    nh.getParam("training_ratio",trainSetSize);
    int temp=SZ;
    nh.getParam("window_size",temp);
    hog.winSize=Size(temp,temp);
    temp=temp/2;
    nh.getParam("block_size",temp);
    hog.blockSize=Size(temp,temp);
    temp=temp/2;
    nh.getParam("block_stride",temp);
    hog.blockStride=Size(temp,temp);
    nh.getParam("cell_size",temp);
    hog.cellSize=Size(temp,temp);
    nh.getParam("nbins",hog.nbins);
    nh.getParam("nlevels",hog.nlevels);
    nh.getParam("gamma",Gamma);
    nh.getParam("c_coefficient",Ccoef);
    nh.getParam("hog_filename",hog_filename);
    nh.getParam("svm_filename",svm_filename);


    // for(int i = 0; i < num_of_dataset; i ++) {
    //     pathNameCones[i].insert(0, pkg_path);
    //     pathNameNonCones[i].insert(0, pkg_path);
    // }
    hog_filename.insert(0,pkg_path);
    svm_filename.insert(0,pkg_path);

    ROS_INFO("configure done");


}

int main(int argc,char** argv){
    ros::init(argc,argv,"sampler");
    ros::NodeHandle nh;
    InitGlobalVariables(nh);


    time_point<std::chrono::steady_clock> start;
    time_point<std::chrono::steady_clock> end;
    milliseconds diff;
    vector<Mat> trainCells;
    vector<Mat> validateCells;
    vector<Mat> testCells;
    vector<int> trainLabels;
    vector<int> validateLabels;
    vector<int> testLabels;

    start = std::chrono::steady_clock::now();
    loadDataLabel(pathNameCones,numCones,pathNameNonCones,numNonCones,trainCells,validateCells,testCells,trainLabels,validateLabels,testLabels);
    end = std::chrono::steady_clock::now();
    diff = duration_cast<milliseconds>(end-start);
    cout<<"traincell: "<<trainCells.size()<<" validatecell: "<<validateCells.size()<<" testcell: "<<testCells.size()<<endl;
    cout<<"loadDataLabel:"<<diff.count()<<" ms"<<endl;

    vector<Mat> deskewedTrainCells;
    vector<Mat> deskewedValidateCells;
    vector<Mat> deskewedTestCells;

    start = std::chrono::steady_clock::now();
    CreateDeskewedData(deskewedTrainCells,deskewedValidateCells,deskewedTestCells,trainCells,validateCells,testCells);
    end = std::chrono::steady_clock::now();
    diff = duration_cast<milliseconds>(end-start);
    cout<<"CreateDeskewedData:"<<diff.count()<<" ms"<<endl;

    std::vector<std::vector<float> > trainHOG;
    std::vector<std::vector<float> > validateHOG;
    std::vector<std::vector<float> > testHOG;

    start = std::chrono::steady_clock::now();
    CreateDataHOG(trainHOG,validateHOG,testHOG,deskewedTrainCells,deskewedValidateCells,deskewedTestCells);
    end = std::chrono::steady_clock::now();
    diff = duration_cast<milliseconds>(end-start);
    cout<<"CreateDataHOG:"<<diff.count()<<" ms"<<endl;

    int descriptor_size = trainHOG[0].size();
    cout<<descriptor_size<<endl;
    Mat trainMat(trainHOG.size(),descriptor_size,CV_32FC1);
    Mat validateMat(validateHOG.size(),descriptor_size,CV_32FC1);
    Mat testMat(testHOG.size(),descriptor_size,CV_32FC1);

    start = std::chrono::steady_clock::now();
    ConvertVectortoMatrix(trainHOG,validateHOG,testHOG,trainMat,validateMat,testMat);
    end = std::chrono::steady_clock::now();
    diff = duration_cast<milliseconds>(end-start);
    cout<<"ConvertVectortoMatrix:"<<diff.count()<<" ms"<<endl;

    Mat testResponse;

    start = std::chrono::steady_clock::now();
    SVMtrain(trainMat,trainLabels,testResponse,testMat);
    end = std::chrono::steady_clock::now();
    diff = duration_cast<milliseconds>(end-start);
    std::cout<<"SVMtrain:"<< diff.count() << " ms" << std::endl;

    float count = 0;
    float accuracy = 0 ;
    start = std::chrono::steady_clock::now();
    SVMevaluate(testResponse,count,accuracy,testLabels);
    end = std::chrono::steady_clock::now();
    diff = duration_cast<milliseconds>(end-start);
    cout<<"SVMevaluate:"<<diff.count()<<" ms"<<endl;

    cout << "Accuracy        : " << accuracy << "%"<< endl;
    return 0;
}
