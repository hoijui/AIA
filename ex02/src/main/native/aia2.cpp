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
static const int iBinThreshold = 98;
//! Number of applications of the erosion operator.
static const int iNumOfErosions = 2;
//! Number of dimensions in the FD.
static const int iFDNormDimensions = 64;
//! Threshold for detection.
static const double dDetectionThreshold = 0.005; //could probably be 0.005
//! Delay before program is resumed after image display.
static const int iImageDelay = 50;

void getContourLine(const Mat& matImg, vector<Mat>& vmatObjList, const int &iThresh, const int &k);
Mat makeFD(const Mat& matContour);
Mat normFD(Mat& fd, const int &n);
void showImage(Mat& img, string win, double dur=-1);
void plotFD(const Mat &fd, const string &win, const double &dDuration=-1);

//! Print out more information about what is going on currently.
#define DEBUG_MODE 0

/**
 * main function, loads and saves image
 * usage:
 * argv[1] path to query image
 * argv[2] example image for class 1
 * argv[3] example image for class 2
 */
int main(int argc,
         char** argv)
{

    // check if image paths were defined
    if (argc != 4)
    {
        cerr << "Usage: aia2 <input image>  <class 1 example>  <class 2 example>" << endl;
        return -1;
    }

    // process image data base
    // load image as gray-scale, paths in argv[2] and argv[3]
    cout << argv[2] << endl;
    Mat matExC1 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );
    Mat matExC2  = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE );
    if ((!matExC1.data) || (!matExC2.data))
    {
        cerr << "ERROR: Cannot load class examples in\n" << argv[2] << "\n" << argv[3] << endl;
        return -1;
    }

    // get contour line from images
    vector<Mat> vmatContourLines1;
    vector<Mat> vmatContourLines2;

    getContourLine(matExC1, vmatContourLines1, iBinThreshold, iNumOfErosions);
    getContourLine(matExC2, vmatContourLines2, iBinThreshold, iNumOfErosions);

    // calculate fourier descriptor
    Mat matFD1 = makeFD(vmatContourLines1.front());
    Mat matFD2 = makeFD(vmatContourLines2.front());

    plotFD(matFD1,"Fourier Descriptor Raw 1",iImageDelay);

    // normalize  fourier descriptor
    Mat matFD1Norm = normFD(matFD1, iFDNormDimensions);
    Mat matFD2Norm = normFD(matFD2, iFDNormDimensions);


    //plot the intermediate results
    plotFD(matFD1Norm,"Fourier Descriptor 1",iImageDelay);
    //plotFD(matFD2Norm,"Fourier Descriptor 2",iImageDelay);


    // process query image
    // load image as gray-scale, path in argv[1]
    Mat matQuery = imread( argv[1], 0);
    if (!matQuery.data)
    {
        cerr << "ERROR: Cannot load query image in\n" << argv[1] << endl;
        return -1;
    }

    // get contour lines from image
    vector<Mat> vmatContourLinesResult;
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
#if DEBUG_MODE
    cout << "Main: Resulting image dimensions are " << matResult.cols << " " << matResult.rows << endl;
#endif

    for (vector<Mat>::const_iterator c = vmatContourLinesResult.begin(); c != vmatContourLinesResult.end(); c++, i++)
    {
        cout << "Main: Checking object candidate no " << i << " :" << endl;

        for (int i = 0; i < c->rows; i++)
        {
            matResult.at<Vec3b>(c->at<Vec2i>(0,i)[1], c->at<Vec2i>(0,i)[0]) = Vec3b(255,0,0);
        }
        showImage(matResult, "Current Object", iImageDelay);

        // if fourier descriptor has too few components (too small contour), then skip it
        if (c->rows < iFDNormDimensions)
        {
            cout << "Main: Too less boundary points: Min " << iFDNormDimensions << " actual " << c->rows << endl;
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
        if (min(err1, err2) > dDetectionThreshold)
        {
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
        for (int i = 0; i < c->rows; i++)
        {
            matResult.at<Vec3b>(c->at<Vec2i>(0,i)[1], c->at<Vec2i>(0,i)[0]) = col;
        }
        // for intermediate results, use the following line
#if DEBUG_MODE
        showImage(matResult, "Current Object", iImageDelay);
#endif
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
Mat normFD(Mat& matFD,
           const int &n)
{
    if(n % 2 != 0)
    {
        cerr << "normFD: Used frequencies must be even, is " << n << "." << endl;
        return matFD;
    }
    else
    {
        //Range rowRange(0,n);
        Mat matOut(n,1,CV_32FC2);
        float fScale = sqrt(pow(matFD.at<Vec2f>(0,1)[0],2)+pow(matFD.at<Vec2f>(0,1)[1],2)); // magnitude of second element

        for(int i = 0; i < n/2; i++)
        {
            matOut.row(i) = matFD.row(i) / fScale;

            matOut.row(n-1-i) = matFD.row(matFD.rows-1-i) / fScale;


            matOut.at<Vec2f>(0,i)[0] = sqrt(pow(matOut.at<Vec2f>(0,i)[0],2) + pow(matOut.at<Vec2f>(0,i)[1],2)); //rotation invariance -> delete phase information
            matOut.at<Vec2f>(0,i)[1] = 0;

            matOut.at<Vec2f>(0,n-1-i)[0] = sqrt(pow(matOut.at<Vec2f>(0,n-1-i)[0],2) + pow(matOut.at<Vec2f>(0,n-1-i)[1],2));
            matOut.at<Vec2f>(0,n-1-i)[1] = 0;
        }
        matOut.at<Vec2f>(0,0)[0] = 0;
        //translation invariance -> first element to 0
        //ignore high frequencies -> delete middle elemenents of vector so that only first and last n/2 elements remain
        //scale invariance -> divide all elements by 2nd element
        return matOut;
    }
}

/**
 * calculates the (unnormalized) fourier descriptor from a list of points
 * @param contour  1xN 2-channel matrix, containing N points (x in first, y in second channel)
 * @return fourier descriptor (not normalized)
 */
Mat makeFD(const Mat& matContour)
{
    Mat fMatContour = matContour.clone();
    if(matContour.type() == CV_32SC2) //if the input is only integer precision, convert
        matContour.convertTo(fMatContour,CV_32FC2);

    Size outSize;
    outSize.width = 1;
    //outSize.height = getOptimalDFTSize(contour.rows - 1); //could be activated for faster processing
    outSize.height = fMatContour.rows;
    Mat tempContour(outSize,
                    fMatContour.type(), //when used with CV_32FC2, some assertion fails
                    Scalar::all(0) //initialize with zeros
                    );
    //copy contour to tempcontour
    tempContour = fMatContour.clone();

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
void getContourLine(const Mat& matImg,
                    vector<Mat>& vmatObjList,
                    const int &iThresh,
                    const int &k)
{
    //image preparation
    threshold(matImg,matImg,iThresh,
              255, //set all points meeting the threshold to 255
              THRESH_BINARY //output is a binary image
              );

    //perform closing --> dilate first, then erode
    dilate(matImg,matImg,
           Mat(), //3x3 square kernel
           Point(-1,-1), //upper left corner
           1);

    erode(matImg,matImg,
          Mat(), //Mat() leads to a 3x3 square kernel
          Point(-1,-1), //upper corner
          k);


    /*namedWindow("test", CV_WINDOW_AUTOSIZE);
    imshow("test", img);
    waitKey(0);*/

    //copy image as it is altered by findContours(..)
    Mat imageCopy;
    imageCopy = matImg.clone();

    findContours(imageCopy,vmatObjList,
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
void plotFD(const Mat& matFD,
            const string &szWindowName,
            const double &dDuration)
{
    Mat matInvFD;
    float fMaxX = -HUGE_VAL; //later image dimensions
    float fMaxY = -HUGE_VAL;
    int iOffsetX = 0; //scaled and rounded offset
    int iOffsetY = 0;

    matInvFD = matFD.clone();
    dft(matInvFD,matInvFD,DFT_INVERSE | DFT_SCALE); //inverse dft

    for(int i = 0; i<matInvFD.rows;i++) //find maximum and minimum image dimensions
    {
        fMaxX = max(matInvFD.at<Vec2f>(0,i)[0],fMaxX);
        fMaxY = max(matInvFD.at<Vec2f>(0,i)[1],fMaxY);
    }

    float fScale =  100.0 / min(fMaxX,fMaxY); //maximum image dimension = 100px
    cout << "plotFD: Scale factor is " << fScale << endl;
    matInvFD *= fScale;

    matInvFD.convertTo(matInvFD,CV_32SC2);

    for(int i = 0; i<matInvFD.rows;i++) //find maximum and minimum image dimensions
    {
        iOffsetX = min(matInvFD.at<Vec2i>(0,i)[0],iOffsetX);
        iOffsetY = min(matInvFD.at<Vec2i>(0,i)[1],iOffsetY);
    }

    Mat img = Mat::zeros(ceil(fMaxY*fScale)-iOffsetY,ceil(fMaxX*fScale)-iOffsetX,CV_8UC3);
#if DEBUG_MODE
    cout << "plotFD: Offsets are (" << iOffsetX << " | " << iOffsetY << "), image dimensions are (" << ceil(fMaxX*fScale)-iOffsetX << " | " << ceil(fMaxY*fScale)-iOffsetY << ")" << endl;
#endif
    for(int i = 0; i<matInvFD.rows;i++)
    {
        //we still may have negative coordiates at this point as we centered our contour around (0,0) --> use the offset
        img.at<Vec3b>(matInvFD.at<Vec2i>(0,i)[1] - iOffsetY, matInvFD.at<Vec2i>(0,i)[0] - iOffsetX) = Vec3b(255,255,255);
    }

    namedWindow(szWindowName);
    imshow(szWindowName,img);
    waitKey(dDuration);

}

/**
 * shows the image
 * @param img	the image to be displayed
 * @param win	the window name
 * @param dur	wait number of ms or until key is pressed
 */
void showImage(Mat& img,
               string win,
               double dur)
{

    // use copy for normalization
    Mat tempDisplay = img.clone();
    if (img.channels() == 1)
    {
        normalize(img, tempDisplay, 0, 255, CV_MINMAX);
    }
    // create window and display omage
    namedWindow(win.c_str(), CV_WINDOW_AUTOSIZE);
    imshow(win.c_str(), tempDisplay);
    // wait
    if (dur >= 0)
    {
        waitKey(dur);
    }
}
