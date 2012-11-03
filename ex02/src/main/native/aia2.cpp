//============================================================================
// Name        : aia2.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : use fourier descriptors to classify leafs in images
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// parameters
//! Threshold for image binarization.
static const int binThreshold = 90;
//! Number of applications of the erosion operator.
static const int numOfErosions = 1;
//! Number of dimensions in the FD.
static const int steps = 34;
//! Threshold for detection.
static const double detThreshold = 0.5;

void getContourLine(Mat& img, vector<Mat>& objList, int thresh, int k);
Mat makeFD(Mat& contour);
Mat normFD(Mat& fd, int n);
void showImage(Mat& img, string win, double dur=-1);
void plotFD(Mat& fd, string win, double dur=-1);

/**
 * main function, loads and saves image
 * usage:
 * argv[1] path to query image
 * argv[2] example image for class 1
 * argv[3] example image for class 2
 */
int main(int argc, char** argv) {

	// check if image paths were defined
	if (argc != 4) {
		cerr << "Usage: aia2 <input image>  <class 1 example>  <class 2 example>" << endl;
		return -1;
	}

	// process image data base
	// load image as gray-scale, paths in argv[2] and argv[3]
	Mat exC1 = imread(argv[2], 0);
	Mat exC2  = imread(argv[3], 0);
	if ((!exC1.data) || (!exC2.data)) {
		cerr << "ERROR: Cannot load class examples in\n" << argv[2] << "\n" << argv[3] << endl;
		return -1;
	}

	// get contour line from images
	vector<Mat> contourLines1;
	vector<Mat> contourLines2;
	// TODO
	//binThreshold = ???;
	//numOfErosions = ???;
	getContourLine(exC1, contourLines1, binThreshold, numOfErosions);
	getContourLine(exC2, contourLines2, binThreshold, numOfErosions);

	// calculate fourier descriptor
	Mat fd1 = makeFD(contourLines1.front());
	Mat fd2 = makeFD(contourLines2.front());

	// normalize  fourier descriptor
	// TODO
	//steps = ???;
	Mat fd1_norm = normFD(fd1, steps);
	Mat fd2_norm = normFD(fd2, steps);

	// process query image
	// load image as gray-scale, path in argv[1]
	Mat query = imread( argv[1], 0);
	if (!query.data){
		cerr << "ERROR: Cannot load query image in\n" << argv[1] << endl;
		return -1;
	}

	// get contour lines from image
	vector<Mat> contourLines;
	// TODO
	//binThreshold = ???;
	//numOfErosions = ???;
	getContourLine(query, contourLines, binThreshold, numOfErosions);

	cout << "Found " << contourLines.size() << " object candidates" << endl;

	// just to visualize classification result
	Mat result(query.rows, query.cols, CV_8UC3);
	vector<Mat> tmp;
	tmp.push_back(query);
	tmp.push_back(query);
	tmp.push_back(query);
	merge(tmp, result);

	// loop through all contours found
	int i=1;
	// TODO
	//detThreshold = ???;
	for (vector<Mat>::iterator c = contourLines.begin(); c != contourLines.end(); c++, i++)	{

		cout << "Checking object candidate no " << i << " :\t";

		for (int i = 0; i < c->cols; i++) {
			result.at<Vec3b>(c->at<Vec2i>(0,i)[1], c->at<Vec2i>(0,i)[0]) = Vec3b(255,0,0);
		}
		showImage(result, "Current Object", 0);

		// if fourier descriptor has too few components (too small contour), then skip it
		if (c->cols < steps*2) {
			cout << "Too less boundary points" << endl;
			continue;
		}

		// calculate fourier descriptor
		Mat fd = makeFD(*c);

		// normalize fourier descriptor
		Mat fd_norm = normFD(fd, steps);

		// compare fourier descriptors
		double err1 = norm(fd_norm, fd1_norm) / steps;
		double err2 = norm(fd_norm, fd2_norm) / steps;

		// if similarity is too small, then reject
		if (min(err1, err2) > detThreshold){
			cout << "No class instance ( " << min(err1, err2) << " )" << endl;
			continue;
		}

		// otherwise: assign color according to class
		Vec3b col;
		if (err1 > err2) {
			col = Vec3b(0,0,255);
			cout << "Class 2 ( " << err2 << " )" << endl;
		} else {
			col = Vec3b(0,255,0);
			cout << "Class 1 ( " << err1 << " )" << endl;
		}
		for (int i = 0; i < c->cols; i++) {
			result.at<Vec3b>(c->at<Vec2i>(0,i)[1], c->at<Vec2i>(0,i)[0]) = col;
		}

		// for intermediate results, use the following line
		showImage(result, "Current Object", 0);
	}
	// save result
	imwrite("result.png", result);
	// show final result
	showImage(result, "Result", 0);

	return 0;
}

/**
 * normalize a given fourier descriptor
 * @param fd  the given fourier descriptor
 * @param n  number of used frequencies (should be even)
 * @param out  the normalized fourier descriptor
 */
Mat normFD(Mat& fd, int n) {

	// TODO
}

/**
 * calculates the (unnormalized) fourier descriptor from a list of points
 * @param contour  1xN 2-channel matrix, containing N points (x in first, y in second channel)
 * @param out  fourier descriptor (not normalized)
 */
Mat makeFD(Mat& contour) {

	// TODO
}

/**
 * calculates the contour line of all objects in an image
 * @param img  the input image
 * @param objList  vector of contours, each represented by a two-channel matrix
 * @param thresh  threshold used to binarize the image
 * @param k  number of applications of the erosion operator
 */
void getContourLine(Mat& img, vector<Mat>& objList, int thresh, int k){

	//image preparation
    threshold(img,img,thresh,
              255, //set all points meeting the threshold to 255
              THRESH_BINARY //output is a binary image
              );

    erode(img,img,
          Mat(), //Mat() leads to a 3x3 square kernel
          Point(-1,-1), //upper corner
          k);

    //copy image as it is altered by findContours(..)
    Mat imageCopy(img);

    findContours(imageCopy,objList,
                 CV_RETR_EXTERNAL, //only outer contours
                 CV_CHAIN_APPROX_NONE //no approximation
                 );
}

/**
 * plot fourier descriptor
 * @param fd  the fourier descriptor to be displayed
 * @param win  the window name
 * @param dur  wait number of ms or until key is pressed
 */
void plotFD(Mat& fd, string win, double dur) {

	// TODO
}

/**
 * shows the image
 * @param img	the image to be displayed
 * @param win	the window name
 * @param dur	wait number of ms or until key is pressed
 */
void showImage(Mat& img, string win, double dur) {

	// use copy for normalization
	Mat tempDisplay = img.clone();
	if (img.channels() == 1) {
		normalize(img, tempDisplay, 0, 255, CV_MINMAX);
	}
	// create window and display omage
	namedWindow(win.c_str(), CV_WINDOW_AUTOSIZE);
	imshow(win.c_str(), tempDisplay);
	// wait
	if (dur >= 0) {
		waitKey(dur);
	}
}
