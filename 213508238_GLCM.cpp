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

GLCM::GLCM()
{
}

GLCM::~GLCM()
{
}

Mat GLCM::ConstructGLCM(Mat image, int orientation, int distance) { //this function takes the image,oritentation and distance and calculates the GLCM

	Mat GLCM_Matrix = Mat::zeros(256, 256, CV_32FC1); //creates a matrix of size 256x256 and populates it with zeros 

	int next_pixel_row = 0;				
	int next_pixel_column = 0;				
	int row_start = 0;						
	int column_start = 0;						
	int row_stop = 0;				
	int column_stop = 0; 

	float energy = 0;					
	float homogeneity = 0;						
	float contrast = 0;					
	float correlation = 0;				
	float entropy = 0.0;				

	switch (orientation) {
	case 0:					//If orientation is 0:
		next_pixel_row = 0;				
		next_pixel_column = distance;				
		row_stop = image.rows;				
		column_stop = image.cols - distance;			
		break;

	case 45:				//If orientation is 45:
		next_pixel_row = -1 * distance;			
		next_pixel_column = distance;				
		row_start = distance;						
		row_stop = image.rows;				
		column_stop = image.cols - distance;			
		break;
	case 90:					//If orientation is 90:
		next_pixel_row = -1 * distance;			
		next_pixel_column = 0;				
		row_start = distance;						
		row_stop = image.rows;				
		column_stop = image.cols;				
		break;
	case 135:								//If orientation is 135:
		next_pixel_row = -1 * distance;			
		next_pixel_column = -1 * distance;			
		row_start = distance;						
		column_start = distance;					
		row_stop = image.rows;				
		column_stop = image.cols;				
		break;
	default:
		cout << "ERROR No such Direct." << endl;
		break;
	}

	//Nested loop that traverses every element in the image matrix. Increments the GLCM at position (x,y)

	for (int i = row_start; i < row_stop; i++)
	{
		for (int j = column_start; j < column_stop; j++)
		{
			GLCM_Matrix.at<int>(image.at<uchar>(i, j), image.at<uchar>(i + next_pixel_row, j + next_pixel_column))++;
		}
	}

	//--------------------Normalizing the GLCM--------------------------------------------------

	float sum = 0;

	for (int i = 0; i < 256; i++)					
		for (int j = 0; j < 256; j++)
			sum += GLCM_Matrix.at<int>(i, j);

	for (int i = 0; i < 256; i++)					
		for (int j = 0; j < 256; j++)
			GLCM_Matrix.at<float>(i, j) = GLCM_Matrix.at<int>(i, j) / sum;

	//-------------------------Calculating means and standard deviations-----------------------------------------------------

	float mean_row[256];				
	float mean_column[256];				
	float standard_deviation_row[256];			
	float standard_deviation_column[256];			

	for (int r = 0; r < 256; r++) { //calculating the sum of rows and columns and storing in the array
		for (int c = 0; c < 256; c++) {
			mean_row[r] += GLCM_Matrix.at<float>(r, c);
			mean_column[c] += GLCM_Matrix.at<float>(r, c);
		}
	}

	for (int r = 0; r < 256; r++) { //calculating the mean
		mean_row[r] /= 256;
		mean_column[r] /= 256;
	}

	for (int r = 0; r < 256; r++) {//calculating the standard deviation
		for (int c = 0; c < 256; c++) {
			standard_deviation_row[r] += pow((GLCM_Matrix.at<float>(r, c) - mean_row[r]), 2);
			standard_deviation_column[c] += pow((GLCM_Matrix.at<float>(r, c) - mean_column[c]), 2);
		}
	}

	for (int r = 0; r < 256; r++) {
		standard_deviation_row[r] = sqrt(standard_deviation_row[r] / (255));
		standard_deviation_column[r] = sqrt(standard_deviation_column[r] / (255));
	}

	//--------------------------Calculating the Haralick features------------------------------------------

	for (int r = 0; r < 256; r++) {
		for (int c = 0; c < 256; c++) {
			energy += GLCM_Matrix.at<float>(r, c) * GLCM_Matrix.at<float>(r, c);
			homogeneity += GLCM_Matrix.at<float>(r, c) / (1 + abs(r - c));
			contrast += GLCM_Matrix.at<float>(r, c) * pow(r - c, 2);
			correlation += (GLCM_Matrix.at<float>(c, r)*(r - mean_row[r])*(c - mean_column[c])) / (standard_deviation_row[r] * standard_deviation_column[c]);								//***************
			entropy += GLCM_Matrix.at<float>(r, c) * (float)(-log(GLCM_Matrix.at<float>(r, c) + 1e-99));
		}
	}

	Mat haralick_vector(1, 5, CV_32FC1);					//Vector to store the haralick features
	 haralick_vector.at<float>(0, 0) = energy;
	 haralick_vector.at<float>(0, 1) = homogeneity;
	 haralick_vector.at<float>(0, 2) = contrast;
	 haralick_vector.at<float>(0, 3) = correlation;
	 haralick_vector.at<float>(0, 4) = entropy;

	return  haralick_vector;

}

void GLCM::GetGLCMTrainingData(cv::String folder, cv::String textname) { //function that trained the ANN 

	vector<cv::String> filenames;

	glob(folder, filenames);// global function
	ofstream data_file_GLCM;
	data_file_GLCM.open(textname);
	cout << "Processing GLCM Data. Training The ANN" << endl;
	for (int i = 1; i <= 1; i++) {
		
		for (int i = 0; i < filenames.size(); i++)
		{
			Mat image_load = imread(filenames[i]);	//Loads image of the ith file name in the filenames vector
			cout << "Processing image : " << filenames[i].substr(filenames[i].find_last_of("/\\") + 1) << endl;	//Displays the folder of the image being processed


			if (!image_load.data)
				cout << "Error! Could not load image" << endl;

			Mat temp = ConstructGLCM(image_load, 0, 1);
			data_file_GLCM << temp.at<float>(0, 0) << "|" << temp.at<float>(0, 1) << "|" << temp.at<float>(0, 2) << "|" << temp.at<float>(0, 3) << "|" << temp.at<float>(0, 4) << "|";
			temp = ConstructGLCM(image_load, 45, 1);
			data_file_GLCM << temp.at<float>(0, 0) << "|" << temp.at<float>(0, 1) << "|" << temp.at<float>(0, 2) << "|" << temp.at<float>(0, 3) << "|" << temp.at<float>(0, 4) << "|";
			temp = ConstructGLCM(image_load, 90, 1);
			data_file_GLCM << temp.at<float>(0, 0) << "|" << temp.at<float>(0, 1) << "|" << temp.at<float>(0, 2) << "|" << temp.at<float>(0, 3) << "|" << temp.at<float>(0, 4) << "|";
			temp = ConstructGLCM(image_load, 135, 1);
			data_file_GLCM << temp.at<float>(0, 0) << "|" << temp.at<float>(0, 1) << "|" << temp.at<float>(0, 2) << "|" << temp.at<float>(0, 3) << "|" << temp.at<float>(0, 4) << "|";
			temp = ConstructGLCM(image_load, 0, 2);
			data_file_GLCM << temp.at<float>(0, 0) << "|" << temp.at<float>(0, 1) << "|" << temp.at<float>(0, 2) << "|" << temp.at<float>(0, 3) << "|" << temp.at<float>(0, 4) << "|";
			temp = ConstructGLCM(image_load, 45, 2);
			data_file_GLCM << temp.at<float>(0, 0) << "|" << temp.at<float>(0, 1) << "|" << temp.at<float>(0, 2) << "|" << temp.at<float>(0, 3) << "|" << temp.at<float>(0, 4) << "|";
			temp = ConstructGLCM(image_load, 90, 2);
			data_file_GLCM << temp.at<float>(0, 0) << "|" << temp.at<float>(0, 1) << "|" << temp.at<float>(0, 2) << "|" << temp.at<float>(0, 3) << "|" << temp.at<float>(0, 4) << "|";
			temp = ConstructGLCM(image_load, 135, 2);
			data_file_GLCM << temp.at<float>(0, 0) << "|" << temp.at<float>(0, 1) << "|" << temp.at<float>(0, 2) << "|" << temp.at<float>(0, 3) << "|" << temp.at<float>(0, 4) << "|";

			data_file_GLCM << endl;
		}
	}
	cout << textname << " Processing Complete. " << textname << " file is closed...\n" << endl;
	data_file_GLCM.close();
}



Mat GLCM::LoadGLCMTrainingData() {//loads training data 
	GetGLCMTrainingData("C:/training-and-test-files/empty_train", "glcm_empty_train.txt");
	GetGLCMTrainingData("C:/training-and-test-files/good_train", "glcm_good_train.txt");
	GetGLCMTrainingData("C:/training-and-test-files/bad_ train", "glcm_bad_train.txt");

	Mat GLCMTrainingDataMatrix(108, 40, CV_32FC1);	//Create a matrix with 108 rows (number of samples) and 40 columns (number of features)
cv:String line = "|";

	ifstream reader("glcm_empty_train.txt");		
	int row = 0;							

	while (!reader.eof())					
	{
		getline(reader, line);				
		int pos = 0;
		int column = 0;

		
		while ((pos = line.find("|")) != std::string::npos) {					
			GLCMTrainingDataMatrix.at<float>(row, column) = stof(line.substr(0, pos));	
			line.erase(0, pos + 1);												
			column++;															
		}
		row++;																	
	}
	row--;																		
																				
	ifstream reader1("glcm_good_train.txt");										
	while (!reader1.eof())
	{
		getline(reader1, line);
		int pos = 0;
		int column = 0;
		while ((pos = line.find("|")) != std::string::npos) {
			GLCMTrainingDataMatrix.at<float>(row, column) = stof(line.substr(0, pos));
			line.erase(0, pos + 1);
			column++;
		}
		row++;
	}
	row--;
	ifstream reader2("glcm_bad_train.txt");										
	while (!reader2.eof())
	{
		getline(reader2, line);
		int pos = 0;
		int column = 0;
		while ((pos = line.find("|")) != std::string::npos) {
			GLCMTrainingDataMatrix.at<float>(row, column) = stof(line.substr(0, pos));
			line.erase(0, pos + 1);
			column++;
		}
		row++;
	}

	reader.close();
	reader1.close();
	reader2.close();

	return GLCMTrainingDataMatrix;									//Returns the training data in a Matrix
}

void GLCM::LoadGLCMTestData(Mat test_data, Mat input, int count) {				//Function that stores data from the input matrix to the test_data matrix
	test_data.at<float>(0, 0 + (5 * count)) = input.at<float>(0, 0);	//Control variable called count is used to control which rows the data is stored into in the test_data matrix
	test_data.at<float>(0, 1 + (5 * count)) = input.at<float>(0, 1);	//When count = 1, the data is stored in the first 5 rows.
	test_data.at<float>(0, 2 + (5 * count)) = input.at<float>(0, 2);	//When count = 2, the data is stored in the second 5 rows (row 6 to 10), and so on
	test_data.at<float>(0, 3 + (5 * count)) = input.at<float>(0, 3);
	test_data.at<float>(0, 4 + (5 * count)) = input.at<float>(0, 4);
}

void GLCM::GLCMPredictions(Mat imageTest, CvANN_MLP* mlp, int &count_good, int &count_bad, int &count_empty) {//the testData predictions are made
	Mat testData(1, 40, CV_32FC1);						//Matrix used to store the test data features

	Mat temp =ConstructGLCM(imageTest, 0, 1);
	LoadGLCMTestData(testData, temp, 0);
	temp =ConstructGLCM(imageTest, 45, 1);
	LoadGLCMTestData(testData, temp, 1);
	temp =ConstructGLCM(imageTest, 90, 1);
	LoadGLCMTestData(testData, temp, 2);
	temp =ConstructGLCM(imageTest, 135, 1);
	LoadGLCMTestData(testData, temp, 3);
	temp =ConstructGLCM(imageTest, 0, 2);
	LoadGLCMTestData(testData, temp, 4);
	temp =ConstructGLCM(imageTest, 45, 2);
	LoadGLCMTestData(testData, temp, 5);
	temp =ConstructGLCM(imageTest, 90, 2);
	LoadGLCMTestData(testData, temp, 6);
	temp =ConstructGLCM(imageTest, 135, 2);
	LoadGLCMTestData(testData, temp, 7);

	Mat output(1, 3, CV_32FC1);
	mlp->predict(testData, output);

	float max = output.at<float>(0, 0);		//the value that has the highest prediction has status good 
	if (max < output.at<float>(0, 1))
		max = output.at<float>(0, 1);
	if (max < output.at<float>(0, 2))
		max = output.at<float>(0, 2);

	if (max == output.at<float>(0, 0))
		count_good++;
	else if (max == output.at<float>(0, 1))
		count_empty++;
	else
		count_bad++;
}

void GLCM::MLP1() {
	

	Mat ANNsize(1, 3, CV_32SC1);
	ANNsize.at<int>(0) = 40;
	ANNsize.at<int>(1) = 16;
	ANNsize.at<int>(2) = 3;


	CvANN_MLP mlp;
	mlp.create(ANNsize, CvANN_MLP::SIGMOID_SYM, 0, 0);

	Mat DataMatSamples = LoadGLCMTrainingData();

	Mat classification;
	classification.create(DataMatSamples.rows, 3, CV_32FC1);


	for (int i = 0; i < 37; i++)//for good classification
	{
		classification.at<int>(i, 0) = 1;
		classification.at<int>(i, 1) = 0;
		classification.at<int>(i, 2) = 0;
	}
	for (int i = 37; i < 72; i++)//empty classification
	{
		classification.at<int>(i, 0) = 0;
		classification.at<int>(i, 1) = 1;
		classification.at<int>(i, 2) = 0;
	}
	for (int i = 72; i < 108; i++)//bad classification
	{
		classification.at<int>(i, 0) = 0;
		classification.at<int>(i, 1) = 0;
		classification.at<int>(i, 2) = 1;
	}


	Mat weights(1, DataMatSamples.rows, CV_32FC1, Scalar::all(1.1));

	mlp.train(DataMatSamples, classification, weights);	//train the ANN with training data, their respective labels and weights

	int count_good = 0, count_bad = 0, count_empty = 0;
	cv::String folder = "C:/training-and-test-files/empty_test";
	cout << "\nTesting Empty Images Using GLCM: " << endl;
	vector<cv::String> empty_filenames;					//Vector stores multiple filenames as strings
	glob(folder, empty_filenames);						//Inputs all the filenames from the folder specified, into the good_filenames vector  

	for (int i = 0; i < empty_filenames.size(); i++)
	{
		Mat image_load = imread(empty_filenames[i]);		//Create a matrix of the image from the filename stored in the vector
		GLCMPredictions(image_load, &mlp, count_good, count_bad, count_empty);		//Calls make_prediction function to predict the image's classification
	}
	cout << count_empty << "True Positives" << endl;
	cout << count_bad+count_good << "False Negatives" << endl;
	count_good = 0; count_bad = 0; count_empty = 0;

	folder = "C:/training-and-test-files/good_test";
	cout << "\nTesting GOOD Images Using GLCM: " << endl;
	vector<cv::String> good_filenames;					
	glob(folder, good_filenames);

	for (int i = 0; i < good_filenames.size(); i++)
	{
		Mat image_load = imread(good_filenames[i]);
		GLCMPredictions(image_load, &mlp, count_good, count_bad, count_empty);		//Calls make_prediction function to predict the image's classification
	}
	cout << count_good << "True Positives" << endl;
	cout << count_bad + count_empty << "False Negatives" << endl;
	count_good = 0; count_bad = 0; count_empty = 0;

	folder = "C:/training-and-test-files/bad_test";
	cout << "\nTesting BAD Images Using GLCM: " << endl;
	vector<cv::String> bad_filenames;					
	glob(folder, bad_filenames);

	for (int i = 0; i < bad_filenames.size(); i++)
	{
		Mat image_load = imread(bad_filenames[i]);
		GLCMPredictions(image_load, &mlp, count_good, count_bad, count_empty);		//Calls make_prediction function to predict the image's classification
	}
	cout << count_bad << "True Positives" << endl;
	cout << count_good + count_empty << "False Negatives" << endl;
	count_good = 0; count_bad = 0; count_empty = 0;

}

