//============================================================================
// Name        : aia1.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : load image using opencv, do something nice, save image
//============================================================================

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// function that performs some kind of (simple) image processing
Mat doSomethingThatMyTutorIsGonnaLike(const char* inputImgFilePath);

// usage: path to image in argv[1]
// main function, loads and saves image
int main(int argc, char** argv) {

  // check if image path was defined
  if (argc != 2){
    cerr << "Usage: aia1 <path_to_image>" << endl;
    return -1;
  }
  
  // window names
  string win1 = string ("Input image colored");
  string win2 = string ("Result grey and optimalized");
  
  // some images
  Mat inputImage, outputImage;
  
  // load image as gray-scale, path in argv[1]
  cout << "Load image: start" << endl;
  inputImage = imread( argv[1]);
  if (!inputImage.data)
    cout << "ERROR: image could not be loaded from " << argv[1] << endl;
  else
    cout << "Load image: done ( " << inputImage.rows << " x " << inputImage.cols << " )" << endl;

  // show input image
  namedWindow( win1.c_str(), CV_WINDOW_AUTOSIZE );
  imshow( win1.c_str(), inputImage );
  waitKey(0);

  // do something (reasonable)
  cout << "Processing: start" << endl;
  outputImage = doSomethingThatMyTutorIsGonnaLike( argv[1] );
  cout << "Processing: done" << endl;

  namedWindow( win2.c_str(), CV_WINDOW_AUTOSIZE );
  imshow( win2.c_str(), outputImage );
  waitKey(0);

  // save result
  imwrite("result_grey_optimized.jpg", outputImage);

  return 0;

}

// function that performs some kind of (simple) image processing
Mat doSomethingThatMyTutorIsGonnaLike(const char* inputImgFilePath){
  Mat grey_image; 
  IplImage *img1 = cvLoadImage(inputImgFilePath, 0);

  // show grey input image
  string win3 = string ("Input image grey");
  
  //save grey input image
  cvSaveImage("result_grey.jpg", img1);

  IplImage* out = cvCreateImage( cvGetSize(img1), IPL_DEPTH_8U, 1 );
  cvShowImage("Result gray", img1);
  // TO DO !!!
  //   cvtColor( img, grey_image, CV_RGB2GRAY );
  //	Rect r(150, 80, 180, 230);
  //	Mat out = img(r);
	//IplImage img1 = img;
  cvEqualizeHist( img1, out );
  //imwrite("result_grey.jpg", img1);
	

  //   cvEqualizeHist( gray_image, out );
  //   cvConvertImage(gray_image, mirrored_gray_image, CV_CVTIMG_FLIP);
  // a too easy example:
  return out;
}
