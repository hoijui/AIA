//============================================================================
// Name        : aia3.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : hough transformation
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// function headers
// functions to be written (see below)
void makeFFTObjectMask(vector<Mat>& templ, double scale, double angle, Mat& fftMask);
vector<Mat> makeObjectTemplate(Mat& templateImage, double sigma, double templateThresh);
vector< vector<Mat> > generalHough(Mat& gradImage, vector<Mat>& templ, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange);
void plotHough(vector< vector<Mat> >& houghSpace);
// given functions (see below)
Mat makeTestImage(Mat& temp, double angle, double scale, double* scaleRange);
Mat rotateAndScale(Mat& temp, double angle, double scale);
Mat calcDirectionalGrad(Mat& image, double sigma);
void showImage(Mat& img, string win, double dur);
void circShift(Mat& in, Mat& out, int dx, int dy);
void findHoughMaxima(vector< vector<Mat> >& houghSpace, double objThresh, vector<Scalar>& objList);
void plotHoughDetectionResult(Mat& testImage, vector<Mat>& templ, vector<Scalar>& objList, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange);

/**
main function
usage:
  loads template and optionally test image; performs general hough transformation in order to find objects in test image
  first case (testing): aia3 <path to template>
  second case (application): aia3 <path to template> <path to testimage>
*/
int main(int argc, char** argv) {

    // check if image paths were defined
    if ( (argc != 2) and (argc != 3) ) {
        cerr << "Usage: aia3 <path to template image> [<path to test image>]" << endl;
        return -1;
    }

    // processing parameter
    double sigma;			// standard deviation of directional gradient kernel
    double templateThresh;		// relative threshold for binarization of the template image
    double objThresh;			// relative threshold for maxima in hough space
    double scaleSteps;			// scale resolution in terms of number of scales to be investigated
    double scaleRange[2];		// scale of angles [min, max]
    double angleSteps;			// angle resolution in terms of number of angles to be investigated
    double angleRange[2];		// range of angles [min, max)

    // load template image as gray-scale, paths in argv[1]
    Mat inputImage = imread( argv[1], 0);
    if (!inputImage.data) {
        cerr << "ERROR: Cannot load template image from\n" << argv[1] << endl;
        return -1;
    }
    // convert 8U to 32F
    Mat templateImage;
    inputImage.convertTo(templateImage, CV_32FC1);

    // show template image
//	showImage(templateImage, "Template image", 0);

    // generate test image
    Mat testImage;
    // if there is no path specified, generate test image from template image
    if (argc == 2) {
        // angle to rotate template image (in radian)
        double testAngle = 30./180.*CV_PI;
        // scale to scale template image
        double testScale = 1.5;
        sigma = 1;
        templateThresh = 0.7;
        objThresh = 0.8; // TODO
        scaleSteps = 20; // TODO
//scaleSteps = 4; // TODO
        scaleRange[0] = 0.5; // TODO
        scaleRange[1] = 2.0; // TODO
        angleSteps = 12; // TODO
        angleRange[0] = 0;
        angleRange[1] = 2*CV_PI;
        // generate test image
        testImage = makeTestImage(templateImage, testAngle, testScale, scaleRange);
    } else {
        // if there was a second file specified
        sigma = 1;
        templateThresh = 0.7; // TODO
        objThresh = 0.2; // TODO
        scaleSteps = 32; // TODO
        scaleRange[0] = 0.5; // TODO
        scaleRange[1] = 2.0; // TODO
        angleSteps = 180; // TODO
        angleRange[0] = 0;
        angleRange[1] = 2*CV_PI;
        // load it
        inputImage = imread( argv[2], 0);
        if (!inputImage.data) {
            cerr << "ERROR: Cannot load test image from\n" << argv[2] << endl;
            return -1;
        }
        // and convert it from 8U to 32F
        inputImage.convertTo(testImage, CV_32FC1);
    }

    // show test image
    cout << "main: Showing test image." << endl;
    showImage(testImage, "testImage", 0);

    // calculate directional gradient of test image as complex numbers (two channel image)
    cout << "main: Calculating directional gradient." << endl;
    Mat gradImage = calcDirectionalGrad(testImage, sigma);

    // generate template from template image
    // templ[0] == binary image
    // templ[0] == directional gradient image
    cout << "main: Calculating object template." << endl;
    vector<Mat> templ = makeObjectTemplate(templateImage, sigma, templateThresh);

    // show binary image
    cout << "main: Showing binary part of template." << endl;
    showImage(templ[0], "Binary part of template", 0);

    // perfrom general hough transformation
    cout << "main: Generating hough space." << endl;
    vector< vector<Mat> > houghSpace = generalHough(gradImage, templ, scaleSteps, scaleRange, angleSteps, angleRange);

    // plot hough space (max over angle- and scale-dimension)
    plotHough(houghSpace);

    // find maxima in hough space
    vector<Scalar> objList;
    findHoughMaxima(houghSpace, objThresh, objList);

    // print found objects on screen
    cout << "Number of objects: " << objList.size() << endl;
    int i=0;
    for (vector<Scalar>::iterator it = objList.begin(); it != objList.end(); it++, i++) {
        cout << i << "\tScale:\t" << (scaleRange[1] - scaleRange[0])/(scaleSteps-1)*(*it).val[0] + scaleRange[0];
        cout << "\tAngle:\t" << ((angleRange[1] - angleRange[0])/(angleSteps)*(*it).val[1] + angleRange[0])/CV_PI*180;
        cout << "\tPosition:\t(" << (*it).val[2] << ", " << (*it).val[3] << " )" << endl;
    }

    // show final detection result
    plotHoughDetectionResult(testImage, templ, objList, scaleSteps, scaleRange, angleSteps, angleRange);

    return 0;
}

/**
  shows hough space, eg. as projection of angle- and scale-dimensions down to a single image
  houghSpace:	the hough space as generated by generalHough(..)
*/
void plotHough(vector< vector<Mat> >& houghSpace) {

    // TODO

    Mat houghImage(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1);

    for (int i = 0; i < houghSpace.size(); ++i) {
        for (int j = 0; j < houghSpace.at(i).size(); ++j) {
            Mat& current = houghSpace.at(i).at(j);
            for (int x = 0; x < current.cols; ++x) {
                for (int y = 0; y < current.rows; ++y) {
                    houghImage.at<float>(x,y) += current.at<float>(x,y);
                }
            }
        }
    }
    normalize(houghImage,houghImage,0,1,CV_MINMAX);
    showImage(houghImage, "Hough Space", 0);
}

/**
  creates the fourier-spectrum of the scaled and rotated template
  templ:	the object template; binary image in templ[0], complex gradient in templ[1]
  scale:	the scale factor to scale the template
  angle:	the angle to rotate the template
  fftMask:	the generated fourier-spectrum of the template
*/
void makeFFTObjectMask(vector<Mat>& templ, double scale, double angle, Mat& fftMask) {

    Mat binMask = templ[0].clone();
    Mat complGrad = templ[1].clone();

    binMask = rotateAndScale(binMask,angle,scale);
    complGrad = rotateAndScale(complGrad,angle,scale);

    //correction of the phase shift
    float imRot = sin(angle);
    float reRot = cos(angle);
    //summation of magnitudes of the complex gradient
    float magnitudeSum = 0.0f;
    for(int i = 0; i < complGrad.cols; i++)
    {
        for(int j = 0; j < complGrad.rows; j++)
        {
            float re = complGrad.at<Vec2f>(i,j)[0];
            float im = complGrad.at<Vec2f>(i,j)[1];
            complGrad.at<Vec2f>(i,j)[0] = re * reRot + im * imRot;
            complGrad.at<Vec2f>(i,j)[1] = im * reRot - re * imRot;

            magnitudeSum += sqrt(pow(complGrad.at<Vec2f>(i,j)[0],2)+pow(complGrad.at<Vec2f>(i,j)[1],2));
        }
    }

    //retain scale invariance
    complGrad /= magnitudeSum;

    //multiply to get object mask, insert directly into padded mask
    Mat bigMask = Mat::zeros(fftMask.rows, fftMask.cols, CV_32FC2);
    for(int i = 0; i < binMask.cols; i++)
    {
        for(int j = 0; j < binMask.rows; j++)
        {
            bigMask.at<Vec2f>(i,j)[0] = binMask.at<float>(i,j)*complGrad.at<Vec2f>(i,j)[0];
            bigMask.at<Vec2f>(i,j)[1] = binMask.at<float>(i,j)*complGrad.at<Vec2f>(i,j)[1];
        }
    }

    //circular shift
    circShift(bigMask,bigMask,-binMask.cols/2,-binMask.rows/2);
    dft(bigMask,fftMask,DFT_COMPLEX_OUTPUT);
}

/**
  computes the hough space of the general hough transform
  gradImage:	the gradient image of the test image
  templ:	the template consisting of binary image and complex-valued directional gradient image
  scaleSteps:	scale resolution
  scaleRange:	range of investigated scales [min, max]
  angleSteps:	angle resolution
  angleRange:	range of investigated angles [min, max)
  return:	the hough space: outer vector over scales, inner vector of angles
*/
vector< vector<Mat> > generalHough(Mat& gradImage, vector<Mat>& templ, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange) {

    // TODO

    vector< vector<Mat> > res;

    double scaleStep = (scaleRange[1] - scaleRange[0]) / scaleSteps;
    double angleStep = (angleRange[1] - angleRange[0]) / angleSteps;

    Mat imgFMask(gradImage.rows, gradImage.cols, CV_32FC2);
    dft(gradImage, imgFMask, DFT_COMPLEX_OUTPUT);

    Mat objFMask(gradImage.rows, gradImage.cols, CV_32FC2);

    for (int scaleI = 0; scaleI < (int)scaleSteps; ++scaleI) {
        double scale = scaleRange[0] + (scaleI * scaleStep);
        vector<Mat> angleRes;
        for (int angleI = 0; angleI < (int)angleSteps; ++angleI) {
            double angle = angleRange[0] + (angleI * angleStep);

            // create scaled and rotated template gradient image
            makeFFTObjectMask(templ, scale, angle, objFMask) ;

            Mat complexHough = Mat::zeros(imgFMask.rows, imgFMask.cols, imgFMask.type());
            mulSpectrums(imgFMask,objFMask,complexHough,
                         0,//no flags
                         true //conjugate objFMask before multiplication
                         );
            //now do the inverse dft
            dft(complexHough,complexHough,DFT_INVERSE | DFT_SCALE);

            // get rid of the phase
            Mat hough(complexHough.cols, complexHough.rows, CV_32FC1);
            for (int i = 0; i < complexHough.cols; ++i) {
                for (int j = 0; j < complexHough.rows; ++j) {
                    hough.at<float>(i, j) = abs(complexHough.at<Vec2f>(i, j)[0]);
                    //hough.at<float>(i,j) = sqrt(pow(complexHough.at<Vec2f>(i, j)[0],2)+pow(complexHough.at<Vec2f>(i, j)[1],2));
                }
            }
            angleRes.push_back(hough);
        }
        res.push_back(angleRes);
    }

    return res;
}

/**
  creates object template from template image
  templateImage:	the template image
  sigma:		standard deviation of directional gradient kernel
  templateThresh:	threshold for binarization of the template image
  return:		the computed template
*/
vector<Mat> makeObjectTemplate(Mat& templateImage, double sigma, double templateThresh) {

    // create x-axis
    cout << 0 << endl;
    Mat complexGradients = calcDirectionalGrad(templateImage, sigma);

    Mat complGrad = complexGradients.clone();

    cout << 1 << endl;
    Mat binaryEdges = Mat::zeros(complGrad.rows,complGrad.cols,CV_32FC1);
    for(int i = 0; i < complGrad.cols; i++)
    {
        for(int j = 0; j < complGrad.rows; j++)
        {
            float re = complGrad.at<Vec2f>(i,j)[0];
            float im = complGrad.at<Vec2f>(i,j)[1];
            binaryEdges.at<float>(i,j) = sqrt(pow(re,2)+pow(im,2));
        }
    }
    cout << 2 << endl;
    float maxGrad = 0.0f;
    for (int i = 0; i < binaryEdges.cols; ++i) {
        for (int j = 0; j < binaryEdges.rows; ++j) {
            if (binaryEdges.at<float>(i, j) > maxGrad) {
                maxGrad = binaryEdges.at<float>(i, j);
            }
        }
    }
    cout << 3 << endl;
    float threshhold = templateThresh * maxGrad;
    cout << 3.5 << endl;
    threshold(binaryEdges, binaryEdges, threshhold,
              255, //set all points meeting the threshold to 255
              THRESH_BINARY //output is a binary image
              );

    cout << 4 << endl;
    vector<Mat> res;
    res.push_back(binaryEdges);
    res.push_back(complGrad);
    cout << 5 << endl;
    return res;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
  computes directional gradients
  image:	the input image
  sigma:	standard deviation of the kernel
  return:	the two-channel gradient image
*/
Mat calcDirectionalGrad(Mat& image, double sigma) {

    // compute kernel size
    int ksize = max(sigma*3,3.);
    if (ksize % 2 == 0) {
        ksize++;
    }
    double mu = ksize / 2.0;

    // generate kernels for x- and y-direction
    double val, sum=0;
    Mat kernel(ksize, ksize, CV_32FC1);
    //Mat kernel_y(ksize, ksize, CV_32FC1);
    for(int i=0; i<ksize; i++) {
        for(int j=0; j<ksize; j++) {
            val  = pow((i+0.5-mu)/sigma,2);
            val += pow((j+0.5-mu)/sigma,2);
            val = exp(-0.5*val);
            sum += val;

            kernel.at<float>(i, j) = -(j+0.5-mu)*val;
        }
    }
    kernel /= sum;

    // use those kernels to compute gradient in x- and y-direction independently
    vector<Mat> grad(2);
    filter2D(image, grad[0], -1, kernel);
    filter2D(image, grad[1], -1, kernel.t());

    // combine both real-valued gradient images to a single complex-valued image
    Mat output;
    merge(grad, output);

    return output;
}

/**
  rotates and scales a given image
  image:	the image to be scaled and rotated
  angle:	rotation angle in radians
  scale:	scaling factor
  return:	transformed image
*/
Mat rotateAndScale(Mat& image, double angle, double scale) {

    // create transformation matrices
    // translation to origin
    Mat T = Mat::eye(3, 3, CV_32FC1);
    T.at<float>(0, 2) = -image.cols/2.0;
    T.at<float>(1, 2) = -image.rows/2.0;
    // rotation
    Mat R = Mat::eye(3, 3, CV_32FC1);
    R.at<float>(0, 0) =  cos(angle);
    R.at<float>(0, 1) = -sin(angle);
    R.at<float>(1, 0) =  sin(angle);
    R.at<float>(1, 1) =  cos(angle);
    // scale
    Mat S = Mat::eye(3, 3, CV_32FC1);
    S.at<float>(0, 0) = scale;
    S.at<float>(1, 1) = scale;
    // combine
    Mat H = R*S*T;

    // compute corners of warped image
    Mat corners(1, 4, CV_32FC2);
    corners.at<Vec2f>(0, 0) = Vec2f(0,0);
    corners.at<Vec2f>(0, 1) = Vec2f(0,image.rows);
    corners.at<Vec2f>(0, 2) = Vec2f(image.cols,0);
    corners.at<Vec2f>(0, 3) = Vec2f(image.cols,image.rows);
    perspectiveTransform(corners, corners, H);

    // compute size of resulting image and allocate memory
    float x_start = min( min( corners.at<Vec2f>(0, 0)[0], corners.at<Vec2f>(0, 1)[0]), min( corners.at<Vec2f>(0, 2)[0], corners.at<Vec2f>(0, 3)[0]) );
    float x_end   = max( max( corners.at<Vec2f>(0, 0)[0], corners.at<Vec2f>(0, 1)[0]), max( corners.at<Vec2f>(0, 2)[0], corners.at<Vec2f>(0, 3)[0]) );
    float y_start = min( min( corners.at<Vec2f>(0, 0)[1], corners.at<Vec2f>(0, 1)[1]), min( corners.at<Vec2f>(0, 2)[1], corners.at<Vec2f>(0, 3)[1]) );
    float y_end   = max( max( corners.at<Vec2f>(0, 0)[1], corners.at<Vec2f>(0, 1)[1]), max( corners.at<Vec2f>(0, 2)[1], corners.at<Vec2f>(0, 3)[1]) );

    // create translation matrix in order to copy new object to image center
    T.at<float>(0, 0) = 1;
    T.at<float>(1, 1) = 1;
    T.at<float>(2, 2) = 1;
    T.at<float>(0, 2) = (x_end - x_start + 1)/2.0;
    T.at<float>(1, 2) = (y_end - y_start + 1)/2.0;

    // change homography to take necessary translation into account
    H = T * H;
    // warp image and copy it to output image
    Mat output;
    warpPerspective(image, output, H, Size(x_end - x_start + 1, y_end - y_start + 1), CV_INTER_LINEAR);

    return output;
}

/**
  generates the test image as a transformed version of the template image
  temp:		the template image
  angle:	rotation angle
  scale:	scaling factor
  scaleRange:	scale range [min,max], used to determine the image size
*/
Mat makeTestImage(Mat& temp, double angle, double scale, double* scaleRange) {

    // rotate and scale template image
    Mat small = rotateAndScale(temp, angle, scale);

    // create empty test image
    Mat testImage = Mat::zeros(temp.rows*scaleRange[1]*2, temp.cols*scaleRange[1]*2, CV_32FC1);
    // copy new object into test image
    Mat tmp;
    Rect roi;
    roi = Rect( (testImage.cols - small.cols)*0.5, (testImage.rows - small.rows)*0.5, small.cols, small.rows);
    tmp = Mat(testImage, roi);
    small.copyTo(tmp);

    return testImage;
}

/**
  shows the detection result of the hough transformation
  testImage:	the test image, where objects were searched (and hopefully found)
  templ:	the template consisting of binary image and complex-valued directional gradient image
  objList:	list of objects as defined by findHoughMaxima(..)
  scaleSteps:	scale resolution
  scaleRange:	range of investigated scales [min, max]
  angleSteps:	angle resolution
  angleRange:	range of investigated angles [min, max)
*/
void plotHoughDetectionResult(Mat& testImage, vector<Mat>& templ, vector<Scalar>& objList, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange) {

    // some matrices to deal with color
    Mat red = testImage.clone();
    Mat green = testImage.clone();
    Mat blue = testImage.clone();
    Mat tmp = Mat::zeros(testImage.rows, testImage.cols, CV_32FC1);

    // scale and angle of current object
    double scale, angle;

    // for all objects
    for(vector<Scalar>::iterator it = objList.begin(); it != objList.end(); it++) {

        // compute scale and angle of current object
        scale = (scaleRange[1] - scaleRange[0])/(scaleSteps-1)*(*it).val[0] + scaleRange[0];
        angle = ((angleRange[1] - angleRange[0])/(angleSteps)*(*it).val[1] + angleRange[0]);

        // use scale and angle in order to generate new binary mask of template
        Mat binMask = rotateAndScale(templ[0], angle, scale);

        // perform boundary checks
        Rect binArea = Rect(0, 0, binMask.cols, binMask.rows);
        Rect imgArea = Rect((*it).val[2]-binMask.cols/2., (*it).val[3]-binMask.rows/2, binMask.cols, binMask.rows);
        if ((*it).val[2]-binMask.cols/2 < 0) {
            binArea.x = abs( (*it).val[2]-binMask.cols/2);
            binArea.width = binMask.cols - binArea.x;
            imgArea.x = 0;
            imgArea.width = binArea.width;
        }
        if ((*it).val[3]-binMask.rows/2 < 0) {
            binArea.y = abs( (*it).val[3]-binMask.rows/2);
            binArea.height = binMask.rows - binArea.y;
            imgArea.y = 0;
            imgArea.height = binArea.height;
        }
        if ((*it).val[2]-binMask.cols/2 + binMask.cols >= tmp.cols) {
            binArea.width = binMask.cols - ((*it).val[2]-binMask.cols/2 + binMask.cols - tmp.cols);
            imgArea.width = binArea.width;
        }
        if ((*it).val[3]-binMask.rows/2 + binMask.rows >= tmp.rows) {
            binArea.height = binMask.rows - ( (*it).val[3]-binMask.rows/2 + binMask.rows - tmp.rows );
            imgArea.height = binArea.height;
        }
        // copy this object instance in new image of correct size
        tmp.setTo(0);
        Mat binRoi = Mat(binMask, binArea);
        Mat imgRoi = Mat(tmp, imgArea);
        binRoi.copyTo(imgRoi);

        // delete found object from original image in order to reset pixel values with red (which are white up until now)
        binMask = 1 - binMask;
        imgRoi = Mat(red, imgArea);
        multiply(imgRoi, binRoi, imgRoi);
        imgRoi = Mat(green, imgArea);
        multiply(imgRoi, binRoi, imgRoi);
        imgRoi = Mat(blue, imgArea);
        multiply(imgRoi, binRoi, imgRoi);

        // change red channel
        red = red + tmp*255;
    }

    // generate color image
    vector<Mat> color;
    color.push_back(blue);
    color.push_back(green);
    color.push_back(red);
    Mat display;
    merge(color, display);

    // display color image
    showImage(display, "result", 0);

    // save color image
    imwrite("detectionResult.png", display);
}

/**
  seeks for local maxima within the hough space
  a local maxima has to be larger than all its 8 spatial neighbors, as well as the largest value at this position for all scales and orientations
  houghSpace:	the computed hough space
  objThresh:	relative threshold for maxima in hough space
  objList:	list of detected objects
*/
void findHoughMaxima(vector< vector<Mat> >& houghSpace, double objThresh, vector<Scalar>& objList) {

    // get maxima over scales and angles
    Mat maxImage = Mat::zeros(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1 );
    for (vector< vector<Mat> >::iterator it = houghSpace.begin(); it != houghSpace.end(); it++) {
        for (vector<Mat>::iterator img = (*it).begin(); img != (*it).end(); img++) {
            max(*img, maxImage, maxImage);
        }
    }
    // get global maxima
    double min, max;
    minMaxLoc(maxImage, &min, &max);

    // define threshold
    double threshold = objThresh * max;

    // spatial non-maxima suppression
    Mat bin = Mat(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1, -1);
    for (int y=0; y<maxImage.rows; y++) {
        for (int x=0; x<maxImage.cols; x++) {
            // init
            bool localMax = true;
            // check neighbors
            for (int i=-1; i<=1; i++) {
                int new_y = y + i;
                if ((new_y < 0) or (new_y >= maxImage.rows)) {
                    continue;
                }
                for (int j=-1; j<=1; j++) {
                    int new_x = x + j;
                    if ((new_x < 0) or (new_x >= maxImage.cols)) {
                        continue;
                    }
                    if (maxImage.at<float>(new_y, new_x) > maxImage.at<float>(y, x)) {
                        localMax = false;
                        break;
                    }
                }
                if (!localMax) {
                    break;
                }
            }
            // check if local max is larger than threshold
            if ((localMax) and (maxImage.at<float>(y, x) > threshold)) {
                bin.at<float>(y, x) = maxImage.at<float>(y, x);
            }
        }
    }

    // loop through hough space after non-max suppression and add objects to object list
    double scale, angle;
    scale = 0;
    for (vector< vector<Mat> >::iterator it = houghSpace.begin(); it != houghSpace.end(); it++, scale++) {
        angle = 0;
        for (vector<Mat>::iterator img = (*it).begin(); img != (*it).end(); img++, angle++) {
            for (int y=0; y<bin.rows; y++) {
                for (int x=0; x<bin.cols; x++) {
                    if ((*img).at<float>(y, x) == bin.at<float>(y, x)) {
                        // create object list entry consisting of scale, angle, and position where object was detected
                        Scalar cur;
                        cur.val[0] = scale;
                        cur.val[1] = angle;
                        cur.val[2] = x;
                        cur.val[3] = y;
                        objList.push_back(cur);
                    }
                }
            }
        }
    }
}

/**
shows the image
img	the image to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void showImage(Mat& img, string win, double dur) {

    // use copy for normalization
    Mat tempDisplay;
    if (img.channels() == 1) {
        normalize(img, tempDisplay, 0, 255, CV_MINMAX);
    } else {
        tempDisplay = img.clone();
    }

    tempDisplay.convertTo(tempDisplay, CV_8UC1);

    // create window and display omage
    namedWindow( win.c_str(), 0);
    imshow(win.c_str(), tempDisplay);
    // wait
    if (dur >= 0) {
        cvWaitKey(dur);
    }
    // be tidy
    destroyWindow(win.c_str());
}

/**
Performes a circular shift in (dx,dy) direction
in	input matrix
out	circular shifted matrix
dx	shift in x-direction
dy	shift in y-direction
*/
void circShift(Mat& in, Mat& out, int dx, int dy) {

    Mat tmp = Mat::zeros(in.rows, in.cols, in.type());

    int x, y, new_x, new_y;

    for (y=0; y<in.rows; y++) {
        // calulate new y-coordinate
        new_y = y + dy;
        if (new_y < 0) {
            new_y = new_y + in.rows;
        }
        if (new_y>=in.rows) {
            new_y = new_y - in.rows;
        }

        for (x=0; x<in.cols; x++) {
            // calculate new x-coordinate
            new_x = x + dx;
            if (new_x < 0) {
                new_x = new_x + in.cols;
            }
            if (new_x >= in.cols) {
                new_x = new_x - in.cols;
            }

            tmp.at<Vec2f>(new_y, new_x) = in.at<Vec2f>(y, x);
        }
    }
    out = tmp;
}
