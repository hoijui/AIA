//============================================================================
// Name        : aia6.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : Loads images, applies PCA, models distribution, and does classification
//============================================================================

#include <iostream>
#include <stdio.h>
#include <limits>

#include <list>
#include <vector>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// function headers
// functions to be written (see below)
Mat calcCompLogL(vector<struct comp*>& model, Mat& features);
Mat calcMixtureLogL(vector<struct comp*>& model, Mat& features);
Mat gmmEStep(vector<struct comp*>& model, Mat& features);
void gmmMStep(vector<struct comp*>& model, Mat& features, Mat& posterior);
// given functions (see below)
void initNewComponent(vector<struct comp*>& model, Mat& features);
void plotGMM(vector<struct comp*>& model, Mat& features);
void trainGMM(Mat& data, int numberOfComponents, vector<struct comp*>& model);
void readImageDatabase(string dataPath, vector<Mat>& db);
void genFeatureProjection(vector<Mat>& imgDB, vector<PCA>& featSpace, int vectLen);

struct comp {Mat mean; Mat covar; double weight;};

// main function, generates random vectors and estimates their distribution
int main(int argc, char** argv) {

    // check if image paths were defined
    if (argc != 1){
	cerr << "Usage: aia6" << endl;
	return -1;
    }
    
    // some parameters
    int vectLen = 25;			// dimensionality of the generated feature vectors (number of principal components)
    int numberOfComponents = 10; 	// number of components (clusters) to use by model
    
    // read training data
    cout << "Reading training data..." << endl;
    vector<Mat> trainImgDB;
    readImageDatabase("./img/train/in/", trainImgDB);
    cout << "Done\n" << endl;
      
    // generate PCA-basis
    cout << "Generate PCA-basis from data:" << endl;
    vector<PCA> featProjection;
    genFeatureProjection(trainImgDB, featProjection, vectLen);
    cout << "Done\n" << endl;

    // this is going to contain the individual models (one per category)
    vector< vector<struct comp*> > models;
    
    // start learning
    cout << "Start learning..." << endl;
    for(int c=0; c<10; c++){
	cout << "\nTrain GMM of category " << c << endl;
	cout << " > Project on principal components of category " << c << " :" << endl;
	Mat fea = Mat(trainImgDB.at(c).rows, vectLen, CV_32FC1);
	featProjection.at(c).project( trainImgDB.at(c), fea );
	fea = fea.t();
	cout << "> Done" << endl;
	
	cout << " > Estimate probability density of category " << c << " :" << endl;
	// train the corresponding mixture model using EM...
	vector<struct comp*> model;
	trainGMM(fea, numberOfComponents, model);
	models.push_back(model);
	cout << "> Done" << endl;
    }
    cout << "Done\n" << endl;
        
    // read testing data
    cout << "Reading test data...:\t";
    vector<Mat> testImgDB;
    readImageDatabase("./img/test/in/" , testImgDB);
    cout << "Done\n" << endl;
    
    cout << "Test GMM: Start" << endl;
    Mat confMatrix = Mat::zeros(10, 10, CV_32FC1);
    int dtr = 0;
    // for each category within the test data
    for(int c_true=0; c_true<10; c_true++){

	// init likelihood
	Mat maxMixLogL = Mat(1, testImgDB.at(c_true).rows, CV_32FC1);
	// estimated class
	Mat est = Mat::zeros(1, testImgDB.at(c_true).rows, CV_8UC1);
	
	for(int c_est=0; c_est<10; c_est++){
	    cout << " > Project on principal components of category " << c_est << " :\t";
	    Mat fea = Mat(testImgDB.at(c_true).rows, vectLen, CV_32FC1);
	    featProjection.at(c_est).project( testImgDB.at(c_true), fea );
	    fea = fea.t();
	    cout << "Done" << endl;
	
	    cout << " > Estimate class likelihood of category " << c_est << " :" << endl;
	    
	    // get data log
	    Mat mixLogL = calcMixtureLogL(models.at(c_est), fea);

	    // compare to current max
	    for(int i=0; i<fea.cols; i++){
		if ( ( maxMixLogL.at<float>(0,i) < mixLogL.at<float>(0,i) ) or (c_est == 0) ){
		    maxMixLogL.at<float>(0,i) = mixLogL.at<float>(0,i);
		    est.at<uchar>(0,i) = c_est;
		}
	    }
	    cout << "Done\n" << endl;
	}
	// make corresponding entry in confusion matrix
	for(int i=0; i<testImgDB.at(c_true).rows; i++){
	    //cout << (int)est.at<uchar>(0,i) << "\t" << c_true << endl;
	    confMatrix.at<float>( c_true, (int)est.at<uchar>(0,i) )++;
	    dtr += (int)est.at<uchar>(0,i) == c_true;
	}
    }
    cout << "Test GMM: Done\n" << endl;
    
    cout << endl << "Confusion matrix:" << endl;
    cout << confMatrix << endl;
    cout << endl << "No of correctly classified:\t" << dtr << endl;
    
    return 0;
}

// Computes the log-likelihood of each feature vector in every component of the supplied mixture model.
/*
model:     	structure containing model parameters
features:  	matrix of feature vectors
return: 	the log-likelihood log(p(x_i|y_i=j))
*/
Mat calcCompLogL(vector<struct comp*>& model, Mat& features){
  
	//cout << "calcCompLogL" << endl;
	int numFeatures = features.cols;
	const int& d = features.rows; // dimensions
	int numComponents = model.size();
	Mat compLogL(numFeatures, numComponents, CV_32FC1);

	for (int i = 0; i < numFeatures; ++i) {
		for (int j = 0; j < numComponents; ++j) {
			const Mat& mu_j = model.at(j)->mean;
			const Mat& sigma_j = model.at(j)->covar;

			const Mat& xCentered_j = features.col(i) - mu_j;
			compLogL.at<float>(i, j) =
					- 0.5f * log(determinant(sigma_j))
					- (d / 2.0f) * log(2 * M_PI)
					- 0.5f * ((Mat)(xCentered_j.t() * sigma_j.inv() * xCentered_j)).at<float>(0, 0);
		}
	}
	/*
	cout << "Rows (compLogL): " << compLogL.rows << endl;
	cout << "Cols (compLogL): " << compLogL.cols << endl;
	*/

	return compLogL;
    
}

// Computes the log-likelihood of each feature by combining the likelihoods in all model components.
/*
model:     structure containing model parameters
features:  matrix of feature vectors
return:	   the log-likelihood of feature number i in the mixture model (the log of the summation of alpha_j p(x_i|y_i=j) over j)
*/
Mat calcMixtureLogL(vector<struct comp*>& model, Mat& features){
 
	/*
	cout << "calcMixtureLogL" << endl;
	cout << "model.size(): " << model.size() << endl;
	cout << "model.0.mean: " << model.at(0)->mean.rows << " " << model.at(0)->mean.cols << ": " << model.at(0)->mean << endl;
	cout << "model.0.covar: " << model.at(0)->covar.rows << " " << model.at(0)->covar.cols << ": " << model.at(0)->covar  << endl;
	cout << "model.0.weight: " << model.at(0)->weight << endl;
	cout << "features: " << features.rows << " " << features.cols << endl;
	*/

	int numFeatures = features.cols;
	int numComponents = model.size();


	// getting all log likelihoods for each feature according to each single component
	//Mat compLogL(numFeatures, numComponents, CV_32FC1);
	Mat compLogL = calcCompLogL(model, features);


	//Mat logGaussianMixtureModel(model.size(), features.cols, features.type());
	Mat logGaussianMixtureModel(numFeatures, 1, features.type());


	for (int i = 0; i < features.cols; ++i) {

		// getting the log_c-scale-factor
		double log_c;
		double alpha_j = model.at(0)->weight;
		double log_alpha_j = log(alpha_j);
		double log_prob_j  = compLogL.at<float>(i, 0);
		log_c = log_alpha_j + log_prob_j;
		for (int j = 0; j < model.size(); ++j) {
			alpha_j = model.at(j)->weight;
			log_alpha_j = log(alpha_j);
			log_prob_j  = compLogL.at<float>(i,j);
			if (log_c < (log_alpha_j + log_prob_j)) {
				log_c = log_alpha_j + log_prob_j;
			}
		}
		double inner_log_sum = 0;
		for (int j = 0; j < model.size(); ++j) {
			alpha_j = model.at(j)->weight;
			log_alpha_j = log(alpha_j);
			log_prob_j = compLogL.at<float>(i,j);
			inner_log_sum += exp(log_alpha_j + log_prob_j - log_c);

			/*const Mat& mu_j = model.at(j)->mean;
			const Mat& sigma_j = model.at(j)->covar;
			const int& d = features.rows; // dimensions

			const Mat& xCentered_j = features.col(i) - mu_j;
			logGaussianMixtureModel.at<float>(i, j) = alpha_j *
					(log(pow(determinant(sigma_j), -0.5) / pow(2 * M_PI, d / 2.0))
					+ log(- 0.5f * ((Mat)(xCentered_j.t() * sigma_j.inv() * xCentered_j)).at<float>(0, 0)));*/
		}
		logGaussianMixtureModel.at<float>(i, 1) = log_c + log(inner_log_sum);
		//cout << "likelihood feature " << i << ": " << exp(logGaussianMixtureModel.at<float>(i, 1)) << endl;
	}
	// TODO

	return logGaussianMixtureModel;
    
}

// Computes the posterior over components (the degree of component membership) for each feature.
/*
model:     	structure containing model parameters
features:  	matrix of feature vectors
return:		the posterior p(y_i=j|x_i)
*/
Mat gmmEStep(vector<struct comp*>& model, Mat& features){

	//cout << "gmmEStep" << endl;
	int numFeatures = features.cols;
	int numComponents = model.size();
	Mat posterior(numFeatures, numComponents, CV_32FC1);
	double alpha_j;
	double log_alpha_j;
	double prob_feature_i;
	double prob_feature_i_componentwise;
	Mat mixLogL = calcMixtureLogL(model, features);
	Mat compLogL = calcCompLogL(model, features);
	for (int i = 0; i < numFeatures; i++){
		prob_feature_i = mixLogL.at<float>(i, 1);
		for (int j = 0; j < numComponents; j++) {
			alpha_j = model.at(j)->weight;
			log_alpha_j = log(alpha_j);
			prob_feature_i_componentwise = compLogL.at<float>(i, j);
			posterior.at<float>(i, j) = exp(log_alpha_j + prob_feature_i_componentwise - prob_feature_i);
		}
	}

	/*for (int j = 0; j < numComponents; j++) {
		cout << " gmmEStep mean : " << model.at(j)->mean << endl;
	}*/

	return posterior;
        
}

// Updates a given model on the basis of posteriors previously computed in the E-Step.
/*
model:     structure containing model parameters, will be updated in-place
	   new model structure in which all parameters have been updated to reflect the current posterior distributions.
features:  matrix of feature vectors
posterior: the posterior p(y_i=j|x_i)
*/
void gmmMStep(vector<struct comp*>& model, Mat& features, Mat& posterior){

	//cout << "gmmMStep" << endl;
	int numFeatures = features.cols;
	int numComponents = model.size();

	//Mat mixLogL = calcMixtureLogL(model, features);
	//Mat compLogL = calcCompLogL(model, features);

	//Mat mu_j;
	double N_j;
	double alpha_j;
	//Mat cov_j;
	for (int j = 0; j < numComponents; j++) {
		/*cout << "component : " << j << endl;
		cout << " old mean : " << model.at(j)->mean << endl;*/

		N_j = 0;
		Mat mu_j = Mat::zeros(features.rows, 1, CV_32FC1); // = model.at(j)->mean;
		Mat cov_j = Mat::zeros(features.rows, features.rows, CV_32FC1); //model.at(j)->covar;

		// more precise...
		for (int i = 0; i < numFeatures; i++) {
			N_j += posterior.at<float>(i, j);
			mu_j += features.col(i)*posterior.at<float>(i, j);
		}
		//cout << " new mean : " << mu_j << endl;

		// faster?
		/*N_j  = sum(posterior.col(j)).val[0];
		mu_j.at<float>(0) = sum(features.row(0)*posterior.col(j)).val[0];
		mu_j.at<float>(1) = sum(features.row(1)*posterior.col(j)).val[0];*/
		if (mu_j.at<float>(0) != 0) {
			mu_j = mu_j / N_j;
		}
		for (int i = 0; i < numFeatures; i++) {
			Mat xCentered_j = features.col(i) - mu_j;
			if (i == 0) {
				cov_j = ((Mat)(xCentered_j * xCentered_j.t())) * posterior.at<float>(i, j);
			} else {
				cov_j += ((Mat)(xCentered_j * xCentered_j.t())) * posterior.at<float>(i, j);
			}
		}
		if (cov_j.at<float>(0, 1) != 0) {
			cov_j = cov_j / N_j;
		}
		alpha_j = N_j / numFeatures;

		model.at(j)->weight = alpha_j;
		model.at(j)->mean = mu_j;
		model.at(j)->covar = cov_j;

		/*cout << " new mean : " << model.at(j)->mean << endl;
		cout << " weight : " << model.at(j)->weight << endl;

		cout << " all means : " << endl;
		for (int k= 0; k < numComponents; k++){
			cout << "     mean " << k << " : " << model.at(k)->mean << endl;
		}
		*/
	}
        
}

/* ***********************
   *** Given Functions ***
   *********************** */

// constructs PCA-space based on all samples of each class
/*
imgDB		image data base; each matrix corresponds to one class; each row to one image
featSpace	the PCA-bases for each class
vectLen		number of principal components to be used
*/
void genFeatureProjection(vector<Mat>& imgDB, vector<PCA>& featSpace, int vectLen){

    int c = 0;
    for(vector<Mat>::iterator cat = imgDB.begin(); cat != imgDB.end(); cat++, c++){
	cout << " > Generate PC of category " << c << " :\t";
      	PCA compPCA = PCA(*cat, Mat(), CV_PCA_DATA_AS_ROW, vectLen);
	featSpace.push_back(compPCA);
	cout << "Done" << endl;
    }
}

// reads image data base from disc
/*
dataPath	path to directory
db		each matrix in this vector corresponds to one class, each row of the matrix corresponds to one image 
*/
void readImageDatabase(string dataPath, vector<Mat>& db){

    // directory delimiter. you might wanna use '\' on windows systems
    char delim = '/';

    char curDir[100];
    db.reserve(10);
    
    int numberOfImages = 0;
    for(int c=0; c<10; c++){

	list<Mat> imgList;
	sprintf(curDir, "%s%c%i%c", dataPath.c_str(), delim, c, delim);
 
	// read directory
	DIR* pDIR;
	struct dirent *entry;
	struct stat s;
	
	stat(curDir,&s);

	// if path is a directory
	if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
	    if( pDIR=opendir(curDir) ){
		// for all entries in directory
		while(entry = readdir(pDIR)){
		    // is current entry a file?
		    stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
		    if ( ( (s.st_mode & S_IFMT ) != S_IFDIR ) and ( (s.st_mode & S_IFMT ) == S_IFREG ) ){
			// if all conditions are fulfilled: load data
			Mat img = imread((curDir + string(entry->d_name)).c_str(), 0);
			img.convertTo(img, CV_32FC3);
			img /= 255.;
			imgList.push_back(img);
			numberOfImages++;
		    }
		}
		closedir(pDIR);
	    }else{
		cerr << "\nERROR: cant open data dir " << dataPath << endl;
		exit(-1);
	    }
	}else{
	    cerr << "\nERROR: provided path does not specify a directory: ( " << dataPath << " )" << endl;
	    exit(-1);
	}
	
	int numberOfImages = imgList.size();
	int numberOfPixPerImg = imgList.front().cols * imgList.front().rows;
	    
	Mat feature = Mat(numberOfImages, numberOfPixPerImg, CV_32FC1);
	
	int i = 0;
	for(list<Mat>::iterator img = imgList.begin(); img != imgList.end(); img++, i++){
	    for(int p = 0; p<numberOfPixPerImg; p++){
		feature.at<float>(i, p) = img->at<float>(p);
	    }
	}
	db.push_back(feature);
    }  
}

// Trains a Gaussian mixture model with a specified number of components on the basis of a given set of feature vectors.
/*
data:     		feature vectors, one vector per column
numberOfComponents:	the desired number of components in the trained model
*/
void trainGMM(Mat& data, int numberOfComponents, vector<struct comp*>& model){

    // the number and dimensionality of feature vectors
    int featureNum = data.cols;
    int featureDim = data.rows;

    // initialize the model with one component and arbitrary parameters
    struct comp* fst = new struct comp();
    fst->weight = 1;
    fst->mean = Mat::zeros(featureDim, 1, CV_32FC1);
    fst->covar = Mat::eye(featureDim, featureDim, CV_32FC1);
    model.push_back(fst);

    // the data-log-likelihood
    double dataLogL[2] = {0,0};
        
    // iteratively add components to the mixture model
    for(int i=1; i<=numberOfComponents; i++){
      
	cout << "Current number of components: " << i << endl;
	
        // the current combined data log-likelihood p(X|Omega)
	Mat mixLogL = calcMixtureLogL(model, data);
        dataLogL[0] = sum(mixLogL).val[0];
	dataLogL[1] = 0.;
	
        // EM iteration while p(X|Omega) increases
	int it = 0;
        while( (dataLogL[0] > dataLogL[1]) or (it == 0) ){
	    it++;
	    //printf("Iteration: %d\t:\t%f\r", it, dataLogL[0]);

	    // E-Step (computes posterior)
            Mat posterior = gmmEStep(model, data);

            // M-Step (updates model parameters)
            gmmMStep(model, data, posterior);
	    
            // update the current p(X|Omega)
            dataLogL[1] = dataLogL[0];
	    mixLogL = calcMixtureLogL(model, data);
	    dataLogL[0] = sum(mixLogL).val[0];
	}
	cout << endl;
	
        // visualize the current model (with i components trained)
        //if (featureDim >= 2){
        //   plotGMM(model, data);
	//}

        // add a new component if necessary
        if (i < numberOfComponents){
            initNewComponent(model, data);    
	}
    }
    
    cout << endl << "**********************************" << endl;
    cout << "Trained model: " << endl;
    for(int i=0; i<model.size(); i++){
	cout << "Component " << i << endl;
	cout << "\t>> weight: " << model.at(i)->weight << endl;
	cout << "\t>> mean: " << model.at(i)->mean << endl;
	cout << "\t>> std: [" << sqrt(model.at(i)->covar.at<float>(0,0)) << ", " << sqrt(model.at(i)->covar.at<float>(1,1)) << "]" << endl;
	cout << "\t>> covar: " << endl;
	cout << model.at(i)->covar << endl;
	cout << endl;
    }
    // wait a last time
    //cin.get();

}

// Adds a new component to the input mixture model by spliting one of the existing components in two parts.
/*
model:		Gaussian Mixture Model parameters, will be updated in-place
features:	feature vectors
*/
void initNewComponent(vector<struct comp*>& model, Mat& features){

    // number of components in current model
    int compNum = model.size();

    // number of features
    int featureNum = features.cols;

    // dimensionality of feature vectors (equals 3 in this exercise)
    int featureDim = features.rows;

    // the largest component is split (this is not a good criterion...)
    int splitComp = 0;
    for(int i=0; i<compNum; i++){
	if (model.at(splitComp)->weight < model.at(i)->weight){
	    splitComp = i;
	}
    }

    // split component 'splitComp' along its major axis
    Mat eVec, eVal;
    eigen(model.at(splitComp)->covar, eVal, eVec);

    Mat devVec = 0.5 * sqrt( eVal.at<float>(0) ) * eVec.row(0).t();

    // create new model structure and compute new mean values, covariances, new component weights...
    struct comp* newModel = new struct comp;
    newModel->weight = 0.5 * model.at(splitComp)->weight;
    newModel->mean = model.at(splitComp)->mean - devVec;
    newModel->covar = 0.25 * model.at(splitComp)->covar;

    // modify the split component
    model.at(splitComp)->weight = 0.5*model.at(splitComp)->weight;
    model.at(splitComp)->mean += devVec;
    model.at(splitComp)->covar *= 0.25;

    // add it to old model
    model.push_back(newModel);

}

// Visualises the contents of a feature space and the associated mixture model.
/*
model: 		parameters of a Gaussian mixture model
features: 	feature vectors

Feature vectors are plotted as blue points
The means of components are indicated by red circles
The eigenvectors of the covariance are indicated by blue ellipses
If the feature space has more than 2 dimensions, only the first two dimensions are visualized.
*/
void plotGMM(vector<struct comp*>& model, Mat& features){

    // size of the plot
    int imSize = 500;
  
    // get scaling factor to scale coordinates
    double max_x=0, max_y=0, min_x=0, min_y=0;
    for(int n=0; n<features.cols; n++){
	if (max_x < features.at<float>(0, n) )
	    max_x = features.at<float>(0, n);
	if (min_x > features.at<float>(0, n) )
	    min_x = features.at<float>(0, n);
	if (max_y < features.at<float>(1, n) )
	    max_y = features.at<float>(1, n);
	if (min_y > features.at<float>(1, n) )
	    min_y = features.at<float>(1, n);
    }  
    double scale = (imSize-1)/max((max_x - min_x), (max_y - min_y));
    // create plot
    Mat plot = Mat(imSize, imSize, CV_8UC3, Scalar(255,255,255) );

    // set feature points
    for(int n=0; n<features.cols; n++){
	plot.at<Vec3b>( ( features.at<float>(0, n) - min_x ) * scale, max( (features.at<float>(1,n)-min_y)*scale, 5.) ) = Vec3b(255,0,0);
    }
    // get ellipse of components
    Mat EVec, EVal;
    for(int i=0; i<model.size(); i++){

	eigen(model.at(i)->covar, EVal, EVec);
	double rAng = atan2(EVec.at<float>(0, 1), EVec.at<float>(0, 0));

	// draw components
	circle(plot,  Point( (model.at(i)->mean.at<float>(1,0)-min_y)*scale, (model.at(i)->mean.at<float>(0,0)-min_x)*scale ), 3, Scalar(0,0,255), 2);
	ellipse(plot, Point( (model.at(i)->mean.at<float>(1,0)-min_y)*scale, (model.at(i)->mean.at<float>(0,0)-min_x)*scale ), Size(EVal.at<float>(1)*scale*3, EVal.at<float>(0)*scale*3), rAng*180/CV_PI, 0, 360, Scalar(0,255,0), 1);

    }
    
    // show plot an abd wait for key
    imshow("Current model", plot);
    waitKey(0);

}
