//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <list>
#include <iostream>
#include <cstring>
#include <ctime>
#include <cassert>


namespace {

// from feature_extraction.cpp
void extract_features(const IplImage *image, CvMemStorage *storage, CvSeq *&keypoints, CvSeq *&descriptors, const bool useExtendedDescriptor)
{
	IplImage *grayImage = NULL;
	if (1 == image->nChannels)
		cvCopy(image, grayImage, NULL);
	else
	{
		grayImage = cvCreateImage(cvGetSize(image), image->depth, 1);
#if defined(__GNUC__)
		if (strcasecmp(image->channelSeq, "RGB") == 0)
#else
		if (_stricmp(image->channelSeq, "RGB") == 0)
#endif
			cvCvtColor(image, grayImage, CV_RGB2GRAY);
#if defined(__GNUC__)
		else if (strcasecmp(image->channelSeq, "BGR") == 0)
#else
		else if (_stricmp(image->channelSeq, "BGR") == 0)
#endif
			cvCvtColor(image, grayImage, CV_BGR2GRAY);
		else
			assert(false);
		grayImage->origin = image->origin;
	}

	//
	const double hessianThreshold = 600;

	cvExtractSURF(grayImage, NULL, &keypoints, &descriptors, storage, cvSURFParams(hessianThreshold, useExtendedDescriptor));
	assert(keypoints->total == descriptors->total);

	cvReleaseImage(&grayImage);
}

void match_using_k_nearest_neighbor(const CvSeq *targetKeypoints, const CvSeq *targetDescriptors, const CvSeq *inputKeypoints, const CvSeq *inputDescriptors, const int descriptorLength, const double distanceThreshold, std::list<std::pair<int,int> > &matchedPairs)
{
	CvMat *targetDescriptorMat = cvCreateMat(targetKeypoints->total, descriptorLength, CV_32FC1);
	CvMat *inputDescriptorMat = cvCreateMat(inputKeypoints->total, descriptorLength, CV_32FC1);
	for (int i = 0; i < targetKeypoints->total; ++i)
	{
		const float *descriptor = (float *)cvGetSeqElem(targetDescriptors, i);
		for (int j = 0; j < descriptorLength; ++j)
			CV_MAT_ELEM(*targetDescriptorMat ,float, i, j) = descriptor[j];
	}
	for (int i = 0; i < inputKeypoints->total; ++i)
	{
		const float *descriptor = (float*)cvGetSeqElem(inputDescriptors, i);
		for (int j = 0; j < descriptorLength; ++j)
			CV_MAT_ELEM(*inputDescriptorMat, float, i, j) = descriptor[j];
	}

	//
	const int keypointCount = targetKeypoints->total;
	const int neighborCount = 1;

	CvMat *matches = cvCreateMat(keypointCount, 1, CV_32SC1);  // m x k set of row indices of matching vectors (referring to matrix passed to cvCreateFeatureTree). contains -1 in some columns if fewer than k neighbors found
	CvMat *distances = cvCreateMat(keypointCount, 1, CV_64FC1);  // m x k matrix of distances to k nearest neighbors

	{
		//CvFeatureTree *featureTree = cvCreateFeatureTree(inputDescriptorMat);
		CvFeatureTree *featureTree = cvCreateKDTree(inputDescriptorMat);

		const int k = neighborCount;  // the number of neighbors to find
		const int emax = 50;  // the maximum number of leaves to visit
		cvFindFeatures(featureTree, targetDescriptorMat, matches, distances, k, emax);

		cvReleaseFeatureTree(featureTree);
	}

	//
	int matchCount = 0;
	for (int i = 0; i < keypointCount; ++i)
	{
		for (int j = 0; j < neighborCount; ++j)
		{
			const int idx = CV_MAT_ELEM(*matches, int, i, j);  // the index of the matched object
			if (-1 == idx) continue;

			//const double dist = cvGetReal2D(distances, i, j);
			const double dist = cvmGet(distances, i, j);  // for the single-channel floating-point matrix
			if (std::fabs(dist) < distanceThreshold)
				matchedPairs.push_back(std::make_pair(i, idx));
		}
	}

	//
	cvReleaseMat(&matches);
	cvReleaseMat(&distances);

	cvReleaseMat(&targetDescriptorMat);
	cvReleaseMat(&inputDescriptorMat);
}

}  // unnamed namespace

void feature_matching()
{
	//const std::string targetImageFileName("opencv_data\\beaver_target.png");
	//const std::string inputImageFileName("opencv_data\\beaver_input.png");
 	const std::string targetImageFileName("opencv_data\\melon_target.png");
	const std::string inputImageFileName("opencv_data\\melon_1.png");
	//const std::string inputImageFileName("opencv_data\\melon_2.png");
	//const std::string inputImageFileName("opencv_data\\melon_3.png");

	const char *targetWindowName = "feature matching: target image";
	const char *inputWindowName = "feature matching: input image";
	cvNamedWindow(targetWindowName, CV_WINDOW_AUTOSIZE);
	cvNamedWindow(inputWindowName, CV_WINDOW_AUTOSIZE);

	//
	const int useExtendedDescriptor = 0;  // 0 means basic descriptors (64 elements each), 1 means extended descriptors (128 elements each)

	CvMemStorage *targetStorage = cvCreateMemStorage(0);
	CvMemStorage *inputStorage = cvCreateMemStorage(0);

	IplImage *targetImage = cvLoadImage(targetImageFileName.c_str());
	CvSeq *targetKeypoints = NULL;
	CvSeq *targetDescriptors = NULL;
	extract_features(targetImage, targetStorage, targetKeypoints, targetDescriptors, useExtendedDescriptor);
	IplImage *inputImage = cvLoadImage(inputImageFileName.c_str());
	CvSeq *inputKeypoints = NULL;
	CvSeq *inputDescriptors = NULL;
	extract_features(inputImage, inputStorage, inputKeypoints, inputDescriptors, useExtendedDescriptor);

	//
	std::list<std::pair<int,int> > matchedPairs;

	const int descriptorLength = useExtendedDescriptor ? 128 : 64;
	const double distanceThreshold = 10.0;  // too sensitive !!!  ==>  need to reject outliers
	match_using_k_nearest_neighbor(targetKeypoints, targetDescriptors, inputKeypoints, inputDescriptors, descriptorLength, distanceThreshold, matchedPairs);

	//
	srand((unsigned int)time(NULL));
	for (std::list<std::pair<int,int> >::iterator it = matchedPairs.begin(); it != matchedPairs.end(); ++it)
	{
		// SURF point
		const CvSURFPoint *targetKeypoint = (CvSURFPoint *)cvGetSeqElem(targetKeypoints, it->first);
		const CvSURFPoint *inputKeypoint = (CvSURFPoint *)cvGetSeqElem(inputKeypoints, it->second);

		// SURF descriptor
		//const float *targetDescriptor = (float *)cvGetSeqElem(targetDescriptors, it->first);
		//const float *inputDescriptor = (float *)cvGetSeqElem(inputDescriptors, it->second);

		const int red = std::rand() % 256;
		const int green = std::rand() % 256;
		const int blue = std::rand() % 256;
		{
			const int radius = targetKeypoint->size / 10;
			const double angle = targetKeypoint->dir * CV_PI / 180.0;
			cvCircle(targetImage, cvPoint(cvRound(targetKeypoint->pt.x), cvRound(targetKeypoint->pt.y)), radius, CV_RGB(red,green,blue), 1, CV_AA, 0);
			cvLine(targetImage, cvPoint(cvRound(targetKeypoint->pt.x), cvRound(targetKeypoint->pt.y)), cvPoint(cvRound(targetKeypoint->pt.x + radius * std::cos(angle)), cvRound(targetKeypoint->pt.y + radius * std::sin(angle))), CV_RGB(red,green,blue), 1, CV_AA, 0);
		}
		{
			const int radius = inputKeypoint->size / 10;
			const double angle = inputKeypoint->dir * CV_PI / 180.0;
			cvCircle(inputImage, cvPoint(cvRound(inputKeypoint->pt.x), cvRound(inputKeypoint->pt.y)), radius, CV_RGB(red,green,blue), 1, CV_AA, 0);
			cvLine(inputImage, cvPoint(cvRound(inputKeypoint->pt.x), cvRound(inputKeypoint->pt.y)), cvPoint(cvRound(inputKeypoint->pt.x + radius * std::cos(angle)), cvRound(inputKeypoint->pt.y + radius * std::sin(angle))), CV_RGB(red,green,blue), 1, CV_AA, 0);
		}
	}

	cvShowImage(targetWindowName, targetImage);
	cvShowImage(inputWindowName, inputImage);
	cvWaitKey();

	//
	cvReleaseImage(&targetImage);
	cvReleaseImage(&inputImage);

	cvClearMemStorage(targetStorage);
	cvClearMemStorage(inputStorage);

	cvDestroyWindow(targetWindowName);
	cvDestroyWindow(inputWindowName);
}
