/******************************************************************************
	KESHAV JEEWANLALL
	213508238
	ENEL4AI - Classification Using Neural Network
	10 September 2018

	Header.h - Contains functions to calculate GLCM and prepare training data

*******************************************************************************/

#pragma once
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\core.hpp>
#include <opencv2\contrib\contrib.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\ml.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <iomanip>

using namespace std;
using namespace cv;

class GLCM {
public:
	GLCM();
	~GLCM();
	Mat ConstructGLCM(Mat, int , int);
	void GetGLCMTrainingData(String, String);
	Mat LoadGLCMTrainingData();
	void LoadGLCMTestData(Mat, Mat, int);
	void GLCMPredictions(Mat, CvANN_MLP*, int &,int &,int &);
	void MLP1();
};

