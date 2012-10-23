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
Mat doSomethingThatMyTutorIsGonnaLike(Mat&);

// usage: path to image in argv[1]
// main function, loads and saves image
int main(int argc, char** argv) {

  // check if image path was defined
  if (argc != 2){
    cerr << "Usage: aia1 <path_to_image>" << endl;
    return -1;
  }
  
  // window names
  string win1 = string ("Input image");
  string win2 = string ("Result");
  
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
  outputImage = doSomethingThatMyTutorIsGonnaLike( inputImage );
  cout << "Processing: done" << endl;

  namedWindow( win2.c_str(), CV_WINDOW_AUTOSIZE );
  imshow( win2.c_str(), outputImage );
  waitKey(0);

  // save result
  imwrite("result.jpg", outputImage);

  return 0;

}

// function that performs some kind of (simple) image processing
Mat doSomethingThatMyTutorIsGonnaLike(Mat& img){
  
  // TO DO !!!
  // a too easy example:
  return img.clone();

}