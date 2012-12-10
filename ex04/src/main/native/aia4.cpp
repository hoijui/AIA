//============================================================================
// Name        : aia4.cpp
// Author      : Ronny Haensch
// Version     : 0.1
// Copyright   : -
// Description : naive bayes
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include <list>

using namespace std;
using namespace cv;

// function headers
// functions to be written (see below)
vector< Mat > bayes(vector< Mat >& likeli, vector< float >& classPrior);
// given functions (see below)
void showImage(Mat& img, string win, double dur=0);
vector<Mat> calculateFeatures(Mat& trainImage);
vector< vector< MatND > > estimateLikelihood(vector<Mat>& features, vector<int>& binSize, vector<Scalar>& fRanges, vector<Mat>& refData);
vector< Mat > naiveBayes(vector<Mat>& features, vector< vector< MatND > >& hist, vector<Scalar>& fRanges);
void normRGB(Mat& image, vector<Mat>& features);
void hsv(Mat& image, vector<Mat>& features);
void classify(vector<Mat>& p, vector<Vec3b>& classColor, Mat& result);

/* usage:
  performs pixel-wise classification by naive bayes
  aia4 <path to train image> <path to test image> <path to first reference image> ... <path to n-th reference image>
*/
// main function
int main(int argc, char** argv) {

	// usage
	if (argc == 2) {
		if ( strcmp(argv[1], "-h")*strcmp(argv[1], "--help") == 0) {
			cout << "Usage:\taia4 <path to train image> <path to test image> <path to first reference image> ... <path to n-th reference image>" << endl;
			cout << "\t\tn\t:\tNumber of classes" << endl;
			return 0;
		}
	}

	/* **********************************
	   *** feature parameter settings ***
	   **********************************  */
	// NOTE: This parameters have to be adapted to the actual classification problem!
	// number of features
	int numberOfFeatures = 3;
	// range (minimal and maximal value) of each feature
	// order of range values has to correspond to order of features (see calculateFeatures(..) )
	vector<Scalar> fRanges;
	fRanges.push_back(Scalar(0,1));
	fRanges.push_back(Scalar(0,1));
	fRanges.push_back(Scalar(0,1));
	// number of bins of the histogram used to represent the corresponding probability density distribution of each feature
	// order of binSize has to correspond to order of features (see calculateFeatures(..) )
	vector<int> binSize;
	binSize.push_back(100);
	binSize.push_back(100);
	binSize.push_back(100);

	/* *****************************************
	   *** reference data parameter settings ***
	   *****************************************  */
	// NOTE: This parameters have to be adapted to the actual classification problem!
	// number of classes
	int numberOfClasses = 4;

	if (numberOfClasses != argc-3) {
		cerr << "Number of classes doesn't match number of arguments, i.e. number of reference images" << endl;
		return -1;
	}

	// class names
	// order of class names has to correspond to order of reference images provided as arguments in argv[c+3]
	vector<string> classNames;
	classNames.push_back("city");
	classNames.push_back("forest");
	classNames.push_back("water");
	classNames.push_back("field");
	// class prior
	// order of prior values has to correspond to order of reference images provided as arguments in argv[c+3]
	vector<float> classPrior;
	classPrior.push_back(1./numberOfClasses);
	classPrior.push_back(1./numberOfClasses);
	classPrior.push_back(1./numberOfClasses);
	classPrior.push_back(1./numberOfClasses);
	// color code for result image
	// order of colors has to correspond to order of reference images provided as arguments in argv[c+3]
	vector<Vec3b> classColor;
	classColor.push_back(Vec3b(  0,   0, 255));
	classColor.push_back(Vec3b(  0, 255,   0));
	classColor.push_back(Vec3b(255,   0,   0));
	classColor.push_back(Vec3b(  0, 255, 255));


	/* ******************
	   *** Processing ***
	   ******************  */

	// Training

	// load train image
	cout << "Training: Start" << endl;
	cout << "Load train image: Start" << endl;
	Mat trainImage = imread( argv[1] );
	trainImage.convertTo(trainImage, CV_32FC1);
	if (!trainImage.data) {
		cout << "ERROR: Cannot load train image from\n" << argv[1] << endl;
		return -1;
	}
	cout << "Load train image: Done" << endl;

	// load reference data
	cout << "Load reference data: Start" << endl;
	vector<Mat> refData;
	for(int c = 0; c < numberOfClasses; c++) {
		refData.push_back(imread( argv[c+3], 0));
		refData[c].convertTo(refData[c], CV_8UC1);
		if (!(refData[c]).data) {
			cout << "ERROR: Cannot load reference image from\n" << argv[c+3] << endl;
			return -1;
		}
	}
	cout << "Load reference data: Done" << endl;

	// calculate features
	cout << "Calculate features: Start" << endl;
	vector<Mat> features = calculateFeatures(trainImage);
	cout << "Calculate features: Done" << endl;

	// estimate individual likelihood distributions
	cout << "Estimate likelihood: Start" << endl;
	vector< vector< MatND > > hist = estimateLikelihood(features, binSize, fRanges, refData);
	cout << "Estimate likelihood: Done" << endl;

	cout << "Training: Done" << endl;

	cout << "\n\n" << endl;

	// Application

	// load test image
	cout << "Application: Start" << endl;
	cout << "Load test image: Start" << endl;
	Mat testImage = imread( argv[2] );
	testImage.convertTo(testImage, CV_32FC1);
	if (!testImage.data) {
		cout << "ERROR: Cannot load test image from\n" << argv[2] << endl;
		return -1;
	}
	cout << "Load test image: Done" << endl;

	// calculate features
	cout << "Calculate features: Start" << endl;
	features.clear();
	features = calculateFeatures(testImage);
	cout << "Calculate features: Done" << endl;

	// calculate likelihood via naive bayes
	cout << "Apply NB: Start" << endl;
	vector< Mat > likeli = naiveBayes(features, hist, fRanges);
	cout << "Apply NB: Done" << endl;

	// maximum likelihood classification
	cout << "Maximum likelihood classification: Start" << endl;
	Mat result_likelihood = Mat::ones(features[0].rows, features[0].cols, CV_8UC3);
	classify(likeli, classColor, result_likelihood);
	imwrite("result_ml.png", result_likelihood);
	cout << "Maximum likelihood classification: Done" << endl;

	// calculate posterior via bayes theorem
	cout << "Apply Bayes Theorem: Start" << endl;
	vector< Mat > posterior = bayes(likeli, classPrior);
	cout << "Apply Bayes Theorem: Done" << endl;

	// maximum posterior classification
	cout << "Maximum posterior classification: Start" << endl;
	Mat result_posterior = Mat::ones(features[0].rows, features[0].cols, CV_8UC3);
	classify(posterior, classColor, result_posterior);
	imwrite("result_map.png", result_posterior);
	cout << "Maximum posterior classification: Done" << endl;

	return 0;
}

// applies Bayes theorem to calculate posterior distribution P(c|x) = p(x|c)*P(c) / p(x)
/*
 likeli	likelihood distribution p(x|c)
 prior	prior distribution P(c)
 return	posterior distribution P(c|x)
*/
vector< Mat > bayes(vector< Mat >& likeli, vector< float >& classPrior) {

	vector<Mat> posterior;

	// TODO
	;

	return posterior;
}

/* ***********************
   *** Given Functions ***
   *********************** */

// applies classification rule: w = argmax_w p(w)
/*
 p			probability (likelihood or posterior)
 classColor		color value assigned to pixel according to estimated class
 result		resulting classification image
*/
void classify(vector<Mat>& p, vector<Vec3b>& classColor, Mat& result) {

	int numberOfClasses = p.size();
	int maxpind = 0;
	// get class of maximal probability...
	for(int y = 0; y < result.rows; y++) {
		for(int x = 0; x < result.cols; x++) {
			for(int c = 0; c < numberOfClasses; c++) {
				if ( p[maxpind].at<float>(y,x) < p[c].at<float>(y,x) )
					maxpind = c;
			}
			// ... and assign color accordingly
			result.at<Vec3b>(y,x) = classColor.at(maxpind);
		}
	}
}

// applies naive bayes to model likelihood distribution
/*
 features	image features
 hist		indiviudal likelihoods of features	p(x_i|c)
 fRanges	range of feature values
 return	joint likelihood p(x|c) = p(x_1|c)*...p(x_n|c)
*/
vector<Mat> naiveBayes(vector<Mat>& features, vector< vector< MatND > >& hist, vector<Scalar>& fRanges) {

	// p(x|c)
	vector<Mat> p;

	int numberOfFeatures = features.size();
	int numberOfClasses = hist.size();

	// p(x_f|c)
	vector< vector< Mat > > backProject;

	// for each pixel get histogram value for each class and feature
	int channels[] = {0};
	for(int c = 0; c < numberOfClasses; c++) {
		backProject.push_back(vector<Mat>());
		p.push_back(Mat::ones(features[0].rows, features[0].cols, CV_32FC1));
		for(int i = 0; i < numberOfFeatures; i++) {
			float range[] = { fRanges[i].val[0], fRanges[i].val[1] };
			const float* ranges[] = { range };
			backProject[c].push_back(Mat());
			// p(x_f|c)
			calcBackProject(&(features[i]), 1, channels, hist[c][i], backProject[c][i], ranges);
			// p(x|c) = p(x_1|c) * ... * p(x_F|c)
			multiply(p[c], backProject[c][i], p[c]);
		}
	}

	return p;
}

// estimates likelihoods p(x_i|c) by a histogram
/*
 features	feature images
 binSize	number of bins
 fRanges	ranges of feature values
 refData	reference data
*/
vector< vector< MatND > > estimateLikelihood(vector<Mat>& features, vector<int>& binSize, vector<Scalar>& fRanges, vector<Mat>& refData) {

	vector< vector< MatND > > hist;

	int numberOfFeatures = features.size();
	int numberOfClasses = refData.size();

	for(int c = 0; c < numberOfClasses; c++) {
		hist.push_back(vector<MatND>());
		for(int i = 0; i < numberOfFeatures; i++) {
			// generate histogram
			hist[c].push_back(MatND());
			int histSize[] = {binSize[i]};
			float range[] = { fRanges[i].val[0], fRanges[i].val[1] };
			const float* ranges[] = { range };
			int channels[] = {0};
			// estimate histogram
			calcHist( &(features[i]), 1, channels, refData[c], hist[c][i], 1, histSize, ranges, true, false );
			// normalize histogram
			hist[c][i] = hist[c][i] / sum(hist[c][i]).val[0];
			// print histogram to screen
			// cout << "class " << c << "\tfeature " << i << "histogram:\n" << hist[c][i] << endl;
		}
	}

	return hist;
}

// calculate image features
/*
 image	the input image
*/
vector<Mat> calculateFeatures(Mat& image) {

	vector<Mat> features;

	// calculate normalized RGB
	normRGB(image, features);
	// calculate HSV
	// hsv(image, features);
	// TODO

	return features;
}

// normalized RGB
void normRGB(Mat& image, vector<Mat>& features) {

	vector<Mat> channels;
	split(image, channels);
	Mat norm = channels[0] + channels[1] + channels[2];
	features.push_back(channels[0] / norm);
	features.push_back(channels[1] / norm);
	features.push_back(channels[2] / norm);
}

// HSV
void hsv(Mat& image, vector<Mat>& features) {

	vector<Mat> channels;
	Mat tmp = image.clone();
	cvtColor(image, tmp, CV_BGR2HSV);
	split(image, channels);
	features.push_back(channels[0] / 360);
	features.push_back(channels[1] / 255);
	features.push_back(channels[2] / 255);
}

// shows the image
/*
img	the image to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void showImage(Mat& img, string win, double dur) {

	// use copy for normalization
	Mat tempDisplay;
	if (img.channels() == 1)
		normalize(img, tempDisplay, 0, 255, CV_MINMAX);
	else
		tempDisplay = img.clone();

	tempDisplay.convertTo(tempDisplay, CV_8UC1);

	// create window and display omage
	namedWindow( win.c_str(), 0 );
	imshow( win.c_str(), tempDisplay );
	// wait
	if (dur>=0) cvWaitKey(dur);
	// be tidy
	destroyWindow(win.c_str());
}
