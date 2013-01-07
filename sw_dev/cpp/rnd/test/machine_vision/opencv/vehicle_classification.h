#ifndef VEHICLE_CLASSIFICATION_H_GUARD
#define VEHICLE_CLASSIFICATION_H_GUARD

#include <sstream>
#include <map>
#include <algorithm>
#include <vector>

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <ml.h>

#include "../controller/configuration.h"

class MVC_Eigencars
{
public:
    MVC_VehicleClassificationParameter* configuration;
    CvMat *car_mean, *car_space;
    CvANN_MLP* neural_network;
    CvSVM* svm;
    IplImage* mask;
    int imsize;
    MVC_Eigencars(MVC_VehicleClassificationParameter* configuration_)
    {
            configuration = configuration_;         

            //Get mask image
            mask = cvCreateImage(cvSize(configuration->norm_size, configuration->norm_size), IPL_DEPTH_8U, 1);     
            ifstream mask_file("mask.txt");
            for (int i=0; i<configuration->norm_size; ++i)
                    for (int j=0; j<configuration->norm_size; ++j)
                    {
                            int val;
                            mask_file >> val;
                            mask->imageData[i*configuration->norm_size+j] = val;
                    }
           
            //Compute car matrix and car mean vector
            imsize = sqr(configuration->norm_size);
            CvMat* car_matrix = cvCreateMat(configuration->num_car, imsize, CV_32FC1);   
            cvZero(car_matrix);     
            CvMat* car_mean_transpose = cvCreateMat(1, imsize, CV_32FC1);   
            cvZero(car_mean_transpose);     
            IplImage* img = cvCreateImage(cvSize(configuration->norm_size, configuration->norm_size), IPL_DEPTH_8U, 1);     
            for (int c=0; c<configuration->num_car; ++c)
            {
                    stringstream filename;
                    int i = c+1;
                    filename << configuration->car_dir << i << ".bmp";             
                    IplImage* car_image = cvLoadImage(filename.str().c_str(), 0);                   
                    cvResize(car_image, img, CV_INTER_NN);
                    cvReleaseImage(&car_image);                     
                    cvEqualizeHist(img, img);
                    for (int r=0; r<imsize; ++r) {
                            cvmSet(car_matrix, c, r, img->imageData[r]);
                            double val = cvmGet(car_mean_transpose, 0, r) + img->imageData[r];
                            cvmSet(car_mean_transpose, 0, r, val/configuration->num_car);
                    }
            }       
            for (int c=0; c<configuration->num_car; ++c)
            {
                    for (int r=0; r<imsize; ++r)
                    {
                            double val = cvmGet(car_matrix, c, r) - cvmGet(car_mean_transpose, 0, r);
                            cvmSet(car_matrix, c, r, val);                 
                    }
            }
           
            CvMat* evalues = cvCreateMat(1, configuration->num_car, CV_32FC1);     
            CvMat* evectors = cvCreateMat(configuration->num_car, imsize, CV_32FC1);                                       
            cvCalcPCA(car_matrix, car_mean_transpose, evalues, evectors, CV_PCA_USE_AVG);                   
            //Eigenvalues area already sorted
           
            car_space = cvCreateMat(configuration->num_eigen, imsize, CV_32FC1);
            cvZero(car_space);
            for (int c=0; c<configuration->num_eigen; ++c)                 
                    for (int r=0; r<imsize; ++r)
                            cvmSet(car_space, c, r, cvmGet(evectors, c, r));                                       
            car_mean = cvCreateMat(imsize, 1, CV_32FC1);
            cvTranspose(car_mean_transpose, car_mean);
            cvReleaseMat(&car_mean_transpose);
            cvReleaseMat(&evectors);
            cvReleaseMat(&evalues);
            cvReleaseImage(&img);

            //Training data
            pair<CvMat*, CvMat*> training_data = mvc_GetTrainingData();

            //Neural network                               
            //mvc_TrainNN(training_data);
            //SVM
            mvc_TrainSVM(training_data);

            cvReleaseMat(&training_data.first);
            cvReleaseMat(&training_data.second);   
    }

        pair<CvMat*, CvMat*> mvc_GetTrainingData();
        void mvc_TrainNN(pair<CvMat*, CvMat*> &training_data);
        void mvc_TrainSVM(pair<CvMat*, CvMat*> &training_data);
        CvMat* mvc_ComputeCarWeight(IplImage* &image); 
};


CvMat*
MVC_Eigencars::mvc_ComputeCarWeight(IplImage* &image)
{               
        assert(image->imageSize == imsize);
        CvMat *col = cvCreateMat(imsize, 1, CV_32FC1); 
        for (int i=0; i<imsize; ++i)
                cvmSet(col, i, 0, image->imageData[i]);
        cvSub(col, car_mean, col);
        CvMat* mult_res = cvCreateMat(configuration->num_eigen, 1, CV_32FC1);
        cvMatMul(car_space, col, mult_res);
        CvMat* result = cvCreateMat(1, configuration->num_eigen, CV_32FC1);
        cvTranspose(mult_res, result);         
        cvReleaseMat(&mult_res);
        cvReleaseMat(&col);
        return result;
}


pair<CvMat*, CvMat*>
MVC_Eigencars::mvc_GetTrainingData()
{
        int ntraining_car = configuration->num_car*2/3;
        int ntraining_noncar = configuration->num_noncar*2/3;
        int ntraining = ntraining_car + ntraining_noncar;
        CvMat* training_inputs = cvCreateMat(ntraining, configuration->num_eigen, CV_32FC1);
        CvMat* training_outputs = cvCreateMat(ntraining, 1, CV_32FC1);
               
        IplImage* img = cvCreateImage(cvSize(configuration->norm_size, configuration->norm_size), IPL_DEPTH_8U, 1);             
        for (int i=1; i<=ntraining_car; ++i)
        {
                //Get the image
                stringstream filename; 
                filename << configuration->car_dir << i << ".bmp";
                IplImage* car_image = cvLoadImage(filename.str().c_str(), 0);   
                cvResize(car_image, img, CV_INTER_NN);
                cvEqualizeHist(img, img);
                cvReleaseImage(&car_image);
                CvMat* weight = mvc_ComputeCarWeight(img);
                for (int j=0; j<configuration->num_eigen; ++j)
                        cvmSet(training_inputs, i-1, j, cvmGet(weight, 0, j));
                cvmSet(training_outputs, i-1, 0, 1);
        }
       
        for (int i=1; i<=ntraining_noncar; ++i)
        {
                //Get the image
                stringstream filename; 
                filename << configuration->noncar_dir << i << ".bmp";
                IplImage* car_image = cvLoadImage(filename.str().c_str(), 0);           
                cvResize(car_image, img, CV_INTER_NN);
                cvReleaseImage(&car_image);                     
                cvEqualizeHist(img, img);
                CvMat* weight = mvc_ComputeCarWeight(img);
                for (int j=0; j<configuration->num_eigen; ++j)
                        cvmSet(training_inputs, ntraining_car + i-1, j, cvmGet(weight, 0, j));
                cvmSet(training_outputs, ntraining_car + i-1, 0, -1);
        }               
        return pair<CvMat*, CvMat*>(training_inputs, training_outputs);
}

void
MVC_Eigencars::mvc_TrainSVM(pair<CvMat*, CvMat*> &data)
{
        CvSVMParams param;     
        param.term_crit.epsilon = 0.01;
        param.term_crit.max_iter = 50;
        param.term_crit.type = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;
        param.svm_type = CvSVM::NU_SVC;
        param.kernel_type = CvSVM::RBF;
        param.gamma = 64;
        param.C = 8;
        param.nu = 0.5;
        svm = new CvSVM();
        svm->train(data.first, data.second, 0, 0, param);
}

void
MVC_Eigencars::mvc_TrainNN(pair<CvMat*, CvMat*> &data)
{
        int a[] = {configuration->num_eigen, 50, 50, 1};
        //3 layers, hidden layer has num_eigen neurons
        CvMat layer_sizes;
        cvInitMatHeader(&layer_sizes, 1, 4, CV_32SC1, a);       
        neural_network = new CvANN_MLP(&layer_sizes, CvANN_MLP::SIGMOID_SYM, 1, 1);
               
        //Create neuron network classification using training examples 
        CvANN_MLP_TrainParams param;   
        param.train_method = CvANN_MLP_TrainParams::BACKPROP;
        param.term_crit.epsilon = 0.01;
        param.term_crit.max_iter = 50;
        param.term_crit.type = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;
        param.bp_dw_scale = 0.1;
        param.bp_moment_scale = 0.1;           
        neural_network->train(data.first, data.second, NULL, 0, param, 0);
}


class MVC_VehicleClassification
{
public:

        MVC_VehicleClassificationParameter* configuration;
        MVC_Eigencars* eigencars;
        MVC_VehicleClassification(MVC_VehicleClassificationParameter* configuration_)
        {
                configuration = configuration_;
                eigencars = new MVC_Eigencars(configuration);           
        }
       
        void mvc_ComputeEigenvehicle(IplImage* &image, vector<MVC_HLPair> &hl_pairs);
};

/*
 * Process a more reliable clue to detect cars with the found pairs
 */
void
MVC_VehicleClassification::mvc_ComputeEigenvehicle(IplImage* &image, vector<MVC_HLPair> &hl_pairs)
{
        for (unsigned int i=0; i<hl_pairs.size(); ++i)
        {
                double x = cvRound((hl_pairs[i].hl1.first + hl_pairs[i].hl2.first)/2.0);
                double y = cvRound((hl_pairs[i].hl1.second + hl_pairs[i].hl2.second)/2.0);
                hl_pairs[i].mvc_SetMidPoint(x, y);
                hl_pairs[i].mvc_SetIsCar(false);

                //Extract a square image from the frame
                int left_bound = (int)(hl_pairs[i].hl1.first - sqr(hl_pairs[i].hlsize));
                int right_bound = (int)(hl_pairs[i].hl1.first + sqr(hl_pairs[i].hlsize));
                int width = right_bound - left_bound;
                int bottom_bound = (int)(max(hl_pairs[i].hl1.second, hl_pairs[i].hl2.second) + 10*hl_pairs[i].hlsize);
                int top_bound = bottom_bound - width;
                IplImage* im = cvCreateImage(cvSize(width, width), IPL_DEPTH_8U, 1);                   
                for (int j=0; j<im->imageSize; ++j)
                        im->imageData[j] = 255;
                for (int c=0; c<width; ++c)
                {
                        for (int r=0; r<width; ++r)
                        {
                                int row = r-1+top_bound;
                                int col = c-1+left_bound;
                                if (mvc_InImage(image, row, col))
                                {                                                                               
                                        int ind = row*image->widthStep + col*image->nChannels;
                                        double val = 0.299*image->imageData[ind]+ 0.5870*image->imageData[ind+1] + 0.114*image->imageData[ind+2];
                                        im->imageData[r*width + c] = cvRound(val);
                                }
                        }
                }
               
                IplImage* im_resize = cvCreateImage(cvSize(configuration->norm_size, configuration->norm_size), IPL_DEPTH_8U, 1);       
                cvResize(im, im_resize, CV_INTER_NN);
                cvReleaseImage(&im);           
                vector<int> values;
                for (int j=0; j<im_resize->imageSize; ++j)
                        if (eigencars->mask->imageData[j] != 0)
                                values.push_back(im_resize->imageData[j]);
                sort(values.begin(), values.end());
                int median = values[(values.size()+1)/2];

                //Must be a car
                if (median < configuration->EC_black_threshold)                 
                        hl_pairs[i].mvc_SetIsCar(true);
               
                //Appearance checking by eigencars analysis
                else if (median < configuration->EC_bright_threshold)
                {
                        cvEqualizeHist(im_resize, im_resize); //equalization                                           
                        CvMat* car_weights = eigencars->mvc_ComputeCarWeight(im_resize);                       
                        CvMat* result = cvCreateMat(1, 1, CV_32FC1);                   
                        //eigencars->neural_network->predict(car_weights, result);
                        //if (cvmGet(result, 0, 0) >= 0)
                        //      hl_pairs[i].mvc_SetIsCar(true); //car is detected
                        if (eigencars->svm->predict(car_weights) >= 0)
                                hl_pairs[i].mvc_SetIsCar(true); //car is detected
                        cvReleaseMat(&car_weights);
                        cvReleaseMat(&result);
                }
                cvReleaseImage(&im_resize);
        }
}

#endif
