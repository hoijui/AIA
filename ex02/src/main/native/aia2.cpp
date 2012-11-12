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
static const int iBinThreshold = 90;
//! Number of applications of the erosion operator.
static const int iNumOfErosions = 3;
//! Number of dimensions in the FD.
static const int iFDNormDimensions = 32;
//! Threshold for detection.
static const double dDetectionThreshold = 0.5;
//! Delay before program is resumed after image display.
static const int iImageDelay = 2000;

void getContourLine(Mat& img, vector<Mat>& objList, int thresh, int k);
Mat makeFD(Mat& contour);
Mat normFD(Mat& fd, int n);
void showImage(Mat& img, string win, double dur=-1);
void plotFD(Mat& fd, string win, double dur=-1);

#define DEBUG_MODE 0

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
    cout << argv[2] << endl;
    Mat matExC1 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );
    Mat matExC2  = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE );
    if ((!matExC1.data) || (!matExC2.data)) {
        cerr << "ERROR: Cannot load class examples in\n" << argv[2] << "\n" << argv[3] << endl;
        return -1;
    }

    // get contour line from images
    vector<Mat> vmatContourLines1;
    vector<Mat> vmatContourLines2;

    getContourLine(matExC1, vmatContourLines1, iBinThreshold, iNumOfErosions);
    getContourLine(matExC2, vmatContourLines2, iBinThreshold, iNumOfErosions);
#if DEBUG_MODE
    //plot contour line
    Mat matTestImage(matExC1.rows, matExC1.cols, CV_8UC3);
    vector<Mat> temp2;
    temp2.push_back(matExC1);
    temp2.push_back(matExC1);
    temp2.push_back(matExC1);
    merge(temp2, matTestImage);

    Mat cont = vmatContourLines1.front();
    //cout << cont.at<Vec2i>(0,1)[1] << " " << cont.at<Vec2i>(0,1)[0] << endl;
    for(int i = 0; i< cont.rows; i++)
    {
        matTestImage.at<Vec3b>(cont.at<Vec2i>(0,i)[1], cont.at<Vec2i>(0,i)[0]) = Vec3b(0,0,255);
    }

    namedWindow("Contour line 1", CV_WINDOW_AUTOSIZE);
    imshow("Contour line 1",matTestImage);
    waitKey(iImageDelay);
#endif


    // calculate fourier descriptor
    Mat matFD1 = makeFD(vmatContourLines1.front());
    Mat matFD2 = makeFD(vmatContourLines2.front());

    // normalize  fourier descriptor
    // TODO
    //steps = ???;
    plotFD(matFD1,"Fourier Descriptor Raw 1",iImageDelay);

    Mat matFD1Norm = normFD(matFD1, iFDNormDimensions);
    Mat matFD2Norm = normFD(matFD2, iFDNormDimensions);


    //plot the intermediate results
    plotFD(matFD1Norm,"Fourier Descriptor 1",iImageDelay);
    plotFD(matFD2Norm,"Fourier Descriptor 2",iImageDelay);


    // process query image
    // load image as gray-scale, path in argv[1]
    Mat matQuery = imread( argv[1], 0);
    if (!matQuery.data){
        cerr << "ERROR: Cannot load query image in\n" << argv[1] << endl;
        return -1;
    }

    // get contour lines from image
    vector<Mat> vmatContourLinesResult;
    // TODO
    //binThreshold = ???;
    //numOfErosions = ???;
    getContourLine(matQuery, vmatContourLinesResult, iBinThreshold, iNumOfErosions);

    cout << "Found " << vmatContourLinesResult.size() << " object candidates" << endl;

    // just to visualize classification result
    Mat matResult(matQuery.rows, matQuery.cols, CV_8UC3);
    vector<Mat> tmp;
    tmp.push_back(matQuery);
    tmp.push_back(matQuery);
    tmp.push_back(matQuery);
    merge(tmp, matResult);

    // loop through all contours found
    int i=1;

    cout << "Main: Resulting image dimensions are " << matResult.cols << " " << matResult.rows << endl;


    // TODO
    //detThreshold = ???;
    for (vector<Mat>::iterator c = vmatContourLinesResult.begin(); c != vmatContourLinesResult.end(); c++, i++)	{

        cout << "Main: Checking object candidate no " << i << " :" << endl;

        for (int i = 0; i < c->rows; i++) {
            matResult.at<Vec3b>(c->at<Vec2i>(0,i)[1], c->at<Vec2i>(0,i)[0]) = Vec3b(255,0,0);
        }
        showImage(matResult, "Current Object", iImageDelay);

        // if fourier descriptor has too few components (too small contour), then skip it
        if (c->rows < iFDNormDimensions*2) {
            cout << "Main: Too less boundary points: Min " << iFDNormDimensions*2 << " actual " << c->rows << endl;
            continue;
        }

        // calculate fourier descriptor
        Mat fd = makeFD(*c);

        // normalize fourier descriptor
        Mat fd_norm = normFD(fd, iFDNormDimensions);

        // compare fourier descriptors
        double err1 = norm(fd_norm, matFD1Norm) / iFDNormDimensions;
        double err2 = norm(fd_norm, matFD2Norm) / iFDNormDimensions;

        // if similarity is too small, then reject
        if (min(err1, err2) > dDetectionThreshold){
            cout << "Main: No class instance ( " << min(err1, err2) << " )" << endl;
            continue;
        }

        // otherwise: assign color according to class
        Vec3b col;
        if (err1 > err2) {
            col = Vec3b(0,0,255);
            cout << "Main: Class 2 ( " << err2 << " )" << endl;
        } else {
            col = Vec3b(0,255,0);
            cout << "Main: Class 1 ( " << err1 << " )" << endl;
        }
        /*cout << "Main: " << endl << *c << endl;
        for (int i = 0; i < c->rows; i++)
        {
            cout << "Main: matResult.at<Vec3b>(" << c->at<Vec2i>(0,i)[1] << "," << c->at<Vec2i>(0,i)[0] << ") =..." << endl;
            matResult.at<Vec3b>(c->at<Vec2i>(0,i)[1], c->at<Vec2i>(0,i)[0]) = col;
        }
        // for intermediate results, use the following line
        showImage(matResult, "Current Object", iImageDelay);*/
    }
    // save result
    imwrite("result.png", matResult);
    // show final result
    showImage(matResult, "Result", 0);

    return 0;
}

/**
 * normalize a given fourier descriptor
 * @param fd  the given fourier descriptor
 * @param n  number of used frequencies (should be even)
 * @return the normalized fourier descriptor
 */
Mat normFD(Mat& fd, int n)
{
    if(n % 2 != 0)
    {
        cerr << "normFD: Used frequencies must be even, is " << n << "." << endl;
        return fd;
    }
    else
    {
        //Range rowRange(0,n);
        Mat out(n,1,CV_32FC2);
        float fScale = sqrt(pow(fd.at<Vec2f>(0,1)[0],2)+pow(fd.at<Vec2f>(0,1)[1],2)); // magnitude of second element

        for(int i = 0; i < n/2; i++)
        {
            out.row(i) = fd.row(i) / fScale;

            out.row(n-1-i) = fd.row(fd.rows-1-i) / fScale;


            out.at<Vec2f>(0,i)[0] = sqrt(pow(out.at<Vec2f>(0,i)[0],2) + pow(out.at<Vec2f>(0,i)[1],2)); //rotation invariance -> delete phase information
            out.at<Vec2f>(0,i)[1] = 0;

            out.at<Vec2f>(0,n-1-i)[0] = sqrt(pow(out.at<Vec2f>(0,n-1-i)[0],2) + pow(out.at<Vec2f>(0,n-1-i)[1],2));
            out.at<Vec2f>(0,n-1-i)[1] = 0;
        }
        out.at<Vec2f>(0,0)[0] = 0;
        //translation invariance -> first element to 0
        //ignore high frequencies -> delete middle elemenents of vector so that only first and last n/2 elements remain
        //scale invariance -> divide all elements by 2nd element
        return out;
    }
}

/**
 * calculates the (unnormalized) fourier descriptor from a list of points
 * @param contour  1xN 2-channel matrix, containing N points (x in first, y in second channel)
 * @return fourier descriptor (not normalized)
 */
Mat makeFD(Mat& contour)
{
    if(contour.type() == CV_32SC2) //if the input is only integer precision, convert
        contour.convertTo(contour,CV_32FC2);

    Size outSize;
    outSize.width = 1;
    //outSize.height = getOptimalDFTSize(contour.rows - 1); //could be activated for faster processing
    outSize.height = contour.rows;
    Mat tempContour(outSize,
                    contour.type(), //when used with CV_32FC2, some assertion fails
                    Scalar::all(0) //initialize with zeros
                    );
    //copy contour to tempcontour
    tempContour = contour.clone();

    dft(tempContour,tempContour);

    return tempContour;
}

/**
 * calculates the contour line of all objects in an image
 * @param img  the input image
 * @param objList  vector of contours, each represented by a two-channel matrix
 * @param thresh  threshold used to binarize the image
 * @param k  number of applications of the erosion operator
 */
void getContourLine(Mat& img, vector<Mat>& objList, int thresh, int k)
{
    //image preparation
    threshold(img,img,thresh,
              255, //set all points meeting the threshold to 255
              THRESH_BINARY //output is a binary image
              );

    erode(img,img,
          Mat(), //Mat() leads to a 3x3 square kernel
          Point(-1,-1), //upper corner
          k);

    /*namedWindow("test", CV_WINDOW_AUTOSIZE);
    imshow("test", img);
    waitKey(0);*/

    //copy image as it is altered by findContours(..)
    Mat imageCopy;
    imageCopy = img.clone();

    findContours(imageCopy,objList,
                 CV_RETR_LIST , //only outer contours
                 CV_CHAIN_APPROX_NONE //no approximation
                 );
}

/**
 * plot fourier descriptor
 * @param fd  the fourier descriptor to be displayed
 * @param win  the window name
 * @param dur  wait number of ms or until key is pressed
 */
void plotFD(Mat& fd, string win, double dur)
{
    Mat invFd;
    float fMaxX = -HUGE_VAL; //later image dimensions
    float fMaxY = -HUGE_VAL;
    float fMinX = HUGE_VAL; //unscaled to correct negative coordinates
    float fMinY = HUGE_VAL;
    int iOffsetX, iOffsetY; //scaled and rounded offset

    invFd = fd.clone();
    dft(invFd,invFd,DFT_INVERSE | DFT_SCALE); //inverse dft

    for(int i = 0; i<invFd.rows;i++) //find maximum and minimum image dimensions
    {
        fMaxX = max(invFd.at<Vec2f>(0,i)[0],fMaxX);
        fMaxY = max(invFd.at<Vec2f>(0,i)[1],fMaxY);
        fMinX = min(invFd.at<Vec2f>(0,i)[0],fMinX);
        fMinY = min(invFd.at<Vec2f>(0,i)[1],fMinY);
    }

    float fScale =  100.0 / min(fMaxX,fMaxY); //maximum image dimension = 100px
    cout << "plotFD: Scale factor is " << fScale << endl;
    invFd *= fScale;

    invFd.convertTo(invFd,CV_32SC2);
    iOffsetX = ceil(fMinX*fScale);
    iOffsetY = ceil(fMinY*fScale);


    cout << "plotFD: " << fMaxX << " " << fMaxY << endl;

    Mat img = Mat::zeros(ceil(fMaxY*fScale)-iOffsetY,ceil(fMaxX*fScale)-iOffsetX,CV_8UC3);
    //cout << "plotFD: Image type = " << fd.type() << " other types are " << CV_32FC2 << endl;
    cout << "plotFD: Offsets are (" << iOffsetX << " | " << iOffsetY << "), image dimensions are (" << ceil(fMaxX*fScale)-iOffsetX << " | " << ceil(fMaxY*fScale)-iOffsetY << ")" << endl;
    for(int i = 0; i<invFd.rows;i++)
    {
        //we still may have negative coordiates at this point as we centered our contour around (0,0) --> use the offset
        cout << "plotFD: Current point is (" << invFd.at<Vec2i>(0,i)[0] << " - " << iOffsetX << " | " << invFd.at<Vec2i>(0,i)[1] << " - " << iOffsetY << ")" << endl;
        img.at<Vec3b>(invFd.at<Vec2i>(0,i)[1] - iOffsetY, invFd.at<Vec2i>(0,i)[0] - iOffsetX) = Vec3b(255,255,255);
    }

    namedWindow(win);
    imshow(win,img);
    waitKey(dur);

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
