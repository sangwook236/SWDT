//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <map>
#include <list>
#include <iostream>
#include <ctime>
#include <cmath>


namespace {
namespace local {

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
#elif defined(_MSC_VER)
		if (_stricmp(image->channelSeq, "RGB") == 0)
#endif
			cvCvtColor(image, grayImage, CV_RGB2GRAY);
#if defined(__GNUC__)
		else if (strcasecmp(image->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
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

// from feature_matching.cpp
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

struct ClusterBinInfo
{
	ClusterBinInfo(const int orientationBinIdx, const int scaleBinIdx, const int xLocationBinIdx, const int yLocationBinIdx)
	: orientationBinIdx_(orientationBinIdx), scaleBinIdx_(scaleBinIdx), xLocationBinIdx_(xLocationBinIdx), yLocationBinIdx_(yLocationBinIdx)
	{
	}
	ClusterBinInfo(const ClusterBinInfo &rhs)
	: orientationBinIdx_(rhs.orientationBinIdx_), scaleBinIdx_(rhs.scaleBinIdx_), xLocationBinIdx_(rhs.xLocationBinIdx_), yLocationBinIdx_(rhs.yLocationBinIdx_)
	{
	}

	bool operator<(const ClusterBinInfo &rhs) const
	{
		return orientationBinIdx_ < rhs.orientationBinIdx_ && scaleBinIdx_ < rhs.scaleBinIdx_;
	}

	int getOrientationBinIndex() const  {  return orientationBinIdx_;  }
	int getScaleBinIndex() const  {  return scaleBinIdx_;  }
	int getXLocationBinIndex() const  {  return xLocationBinIdx_;  }
	int getYLocationBinIndex() const  {  return yLocationBinIdx_;  }

private:
	const int orientationBinIdx_;
	const int scaleBinIdx_;
	const int xLocationBinIdx_;
	const int yLocationBinIdx_;
};

void remove_outlier_using_hough_transform_voting(const CvSeq *targetKeypoints, const CvSeq *targetDescriptors, const CvSeq *inputKeypoints, const CvSeq *inputDescriptors, const int descriptorLength, const std::list<std::pair<int,int> > &matchedPairs, std::map<ClusterBinInfo, std::list<int> > &clusterBins)
{
	int k = 0;
	for (std::list<std::pair<int,int> >::const_iterator it = matchedPairs.begin(); it != matchedPairs.end(); ++it, ++k)
	{
		// SURF point
		const CvSURFPoint *targetKeypoint = (CvSURFPoint *)cvGetSeqElem(targetKeypoints, it->first);
		const CvSURFPoint *inputKeypoint = (CvSURFPoint *)cvGetSeqElem(inputKeypoints, it->second);

		const double angle = inputKeypoint->dir - targetKeypoint->dir > 0.0 ? inputKeypoint->dir - targetKeypoint->dir : 360.0 - (inputKeypoint->dir - targetKeypoint->dir);  // [deg]
		const double scale = double(inputKeypoint->size) / double(targetKeypoint->size);

		const int orientationBinIdx = (int)std::floor(angle * 0.1);  // a bin size of 10 degrees for orientation
		const int scaleBinIdx = (int)std::floor(std::log(scale) / std::log(2.0));  // a factor of 2 for scale, val = 2^n

		{
			const ClusterBinInfo info(orientationBinIdx, scaleBinIdx, 0, 0);
			std::map<ClusterBinInfo, std::list<int> >::iterator mapIt = clusterBins.find(info);
			if (clusterBins.end() == mapIt)
			{
				const std::pair<std::map<ClusterBinInfo, std::list<int> >::iterator, bool> result = clusterBins.insert(std::make_pair(info, std::list<int>()));
				result.first->second.push_back(k);
			}
			else
				mapIt->second.push_back(k);
		}
	}
}

void remove_outlier_using_ransac(const CvSeq *targetKeypoints, const CvSeq *targetDescriptors, const CvSeq *inputKeypoints, const CvSeq *inputDescriptors, const int descriptorLength, const std::list<std::pair<int,int> > &matchedPairs)
{
	// TODO [add] >>
}

void match_features_using_ransac()
{
 	const std::string targetImageFileName("../data/machine_vision/opencv/melon_target.png");
	//const std::string inputImageFileName("../data/machine_vision/opencv/melon_1.png");
	const std::string inputImageFileName("../data/machine_vision/opencv/melon_2.png");
	//const std::string inputImageFileName("../data/machine_vision/opencv/melon_3.png");

	const char *targetWindowName = "outlier removal: target image";
	const char *inputWindowName = "outlier removal: input image";
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
	const double distanceThreshold = 0.4;  // too sensitive !!!  ==>  need to reject outliers
	match_using_k_nearest_neighbor(targetKeypoints, targetDescriptors, inputKeypoints, inputDescriptors, descriptorLength, distanceThreshold, matchedPairs);

	std::map<ClusterBinInfo, std::list<int> > clusterBins;
	remove_outlier_using_hough_transform_voting(targetKeypoints, targetDescriptors, inputKeypoints, inputDescriptors, descriptorLength, matchedPairs, clusterBins);
	//remove_outlier_using_ransac(targetKeypoints, targetDescriptors, inputKeypoints, inputDescriptors, descriptorLength, matchedPairs);

	//
	{
		std::cout << "cluster count: " << clusterBins.size() << std::endl;
		for (std::map<ClusterBinInfo, std::list<int> >::iterator it = clusterBins.begin(); it != clusterBins.end(); ++it)
		{
			std::cout << it->second.size() << ": ( " << it->first.getOrientationBinIndex() << ", " << it->first.getScaleBinIndex() << ", " << it->first.getXLocationBinIndex() << ", " << it->first.getYLocationBinIndex() << " ) => ";
			for (std::list<int>::iterator iter = it->second.begin(); iter != it->second.end(); ++iter)
				std::cout << *iter << ", ";
			std::cout << std::endl;
		}

#if 1
		std::map<ClusterBinInfo, std::list<int> >::iterator maxBinIt = clusterBins.end();
		size_t maxBinSize = 0;
		for (std::map<ClusterBinInfo, std::list<int> >::iterator it = clusterBins.begin(); it != clusterBins.end(); ++it)
			if (maxBinSize < it->second.size())
			{
				maxBinSize = it->second.size();
				maxBinIt = it;
			}

		if (clusterBins.end() == maxBinIt)
		{
			std::cout << "bin size error !!!" << std::endl;
			return;
		}
#else
		std::map<ClusterBinInfo, std::list<int> >::iterator maxBinIt = clusterBins.begin();
		std::advance(maxBinIt, 3);
#endif

		//
		//for (std::list<std::pair<int,int> >::iterator it = matchedPairs.begin(); it != matchedPairs.end(); ++it)
		std::list<std::pair<int,int> >::iterator iterBegin = matchedPairs.begin(), iter;
		for (std::list<int>::iterator it = maxBinIt->second.begin(); it != maxBinIt->second.end(); ++it)
		{
			iter = iterBegin;
			std::advance(iter, *it);

			// SURF point
			const CvSURFPoint *targetKeypoint = (CvSURFPoint *)cvGetSeqElem(targetKeypoints, iter->first);
			const CvSURFPoint *inputKeypoint = (CvSURFPoint *)cvGetSeqElem(inputKeypoints, iter->second);

			// SURF descriptor
			//const float *targetDescriptor = (float *)cvGetSeqElem(targetDescriptors, iter->first);
			//const float *inputDescriptor = (float *)cvGetSeqElem(inputDescriptors, iter->second);

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
	}

	//
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

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void outlier_removal()
{
	local::match_features_using_ransac();
}

}  // namespace my_opencv
