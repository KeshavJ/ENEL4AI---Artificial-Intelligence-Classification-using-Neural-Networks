/**********************************************************
	KESHAV JEEWANLALL
	213508238
	ENEL4AI - Classification using Neural Network
	10 September 2018

***********************************************************/

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
#include "Header.h"

using namespace cv;
using namespace std;



int main()
{
	GLCM getGLCM;							
	getGLCM.MLP1();
	system("pause");
	return 0;
}