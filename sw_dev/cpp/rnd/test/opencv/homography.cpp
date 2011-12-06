#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <list>
#include <limits>
#include <iostream>
#include <ctime>
#include <cassert>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#if defined(max)
#undef max
#endif
#if defined(min)
#undef min
#endif

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
		if (_stricmp(image->channelSeq, "RGB") == 0)
			cvCvtColor(image, grayImage, CV_RGB2GRAY);
		else if (_stricmp(image->channelSeq, "BGR") == 0)
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

bool calc_homography(const CvSeq *keypoints1, const CvSeq *keypoints2, const std::list<std::pair<int,int> > &matchedPairs, CvMat *homography)
{
	const int matchPairCount = (int)matchedPairs.size();
	if (matchPairCount < 4) return false;

	CvMat *pointMat1 = cvCreateMat(2, matchPairCount, CV_32FC1);
	CvMat *pointMat2 = cvCreateMat(2, matchPairCount, CV_32FC1);
	int k = 0;
	for (std::list<std::pair<int,int> >::const_iterator it = matchedPairs.begin(); it != matchedPairs.end(); ++it, ++k)
	{
		const CvSURFPoint *keypoint1 = (CvSURFPoint *)cvGetSeqElem(keypoints1, it->first);
		const CvSURFPoint *keypoint2 = (CvSURFPoint *)cvGetSeqElem(keypoints2, it->second);

		CV_MAT_ELEM(*pointMat1, float, 0, k) = keypoint1->pt.x;
		CV_MAT_ELEM(*pointMat1, float, 1, k) = keypoint1->pt.y;
		CV_MAT_ELEM(*pointMat2, float, 0, k) = keypoint2->pt.x;
		CV_MAT_ELEM(*pointMat2, float, 1, k) = keypoint2->pt.y;
	}

	const int method = CV_RANSAC;  // the method used to computed homography matrix: 0, CV_RANSAC, CV_LMEDS
	const double ransacReprojThreshold = 5;
	if (!cvFindHomography(pointMat1, pointMat2, homography, method, ransacReprojThreshold, NULL))
		return false;

	cvReleaseMat(&pointMat1);
	cvReleaseMat(&pointMat2);

	return true;
}

}  // unnamed namespace

void homography()
{
 	const std::string targetImageFileName("opencv_data\\melon_target.png");
	const std::string inputImageFileName("opencv_data\\melon_1.png");
	//const std::string inputImageFileName("opencv_data\\melon_2.png");
	//const std::string inputImageFileName("opencv_data\\melon_3.png");

	const char *targetWindowName = "homography: target image";
	const char *inputWindowName = "homography: input image";
	const char *resultantWindowName = "homography: resultant image";
	cvNamedWindow(targetWindowName, CV_WINDOW_AUTOSIZE);
	cvNamedWindow(inputWindowName, CV_WINDOW_AUTOSIZE);
	cvNamedWindow(resultantWindowName, CV_WINDOW_AUTOSIZE);

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

	IplImage *resultImage = NULL;

	//
	std::list<std::pair<int,int> > matchedPairs;

	const int descriptorLength = useExtendedDescriptor ? 128 : 64;
	const double distanceThreshold = 0.15;  // too sensitive !!!  ==>  need to reject outliers
	match_using_k_nearest_neighbor(targetKeypoints, targetDescriptors, inputKeypoints, inputDescriptors, descriptorLength, distanceThreshold, matchedPairs);

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

	//
	CvMat *homography = cvCreateMat(3, 3, CV_64FC1);
	cvSetZero(homography);

	if (!calc_homography(targetKeypoints, inputKeypoints, matchedPairs, homography))
	{
		std::cout << "homography cannot be calculated !!!" << std::endl;
		return;
	}

	// planar homography
	{
		const int cornerCount = 4;
		CvMat *A = cvCreateMat(3, cornerCount, CV_64FC1);
		CvMat *B = cvCreateMat(3, cornerCount, CV_64FC1);
		CV_MAT_ELEM(*A, double, 0, 0) = 0;
		CV_MAT_ELEM(*A, double, 1, 0) = 0;
		CV_MAT_ELEM(*A, double, 2, 0) = 1;
		CV_MAT_ELEM(*A, double, 0, 1) = targetImage->width;
		CV_MAT_ELEM(*A, double, 1, 1) = 0;
		CV_MAT_ELEM(*A, double, 2, 1) = 1;
		CV_MAT_ELEM(*A, double, 0, 2) = targetImage->width;
		CV_MAT_ELEM(*A, double, 1, 2) = targetImage->height;
		CV_MAT_ELEM(*A, double, 2, 2) = 1;
		CV_MAT_ELEM(*A, double, 0, 3) = 0;
		CV_MAT_ELEM(*A, double, 1, 3) = targetImage->height;
		CV_MAT_ELEM(*A, double, 2, 3) = 1;

		cvMatMul(homography, A, B);

		CvPoint pt1, pt2;
		pt1.x = cvRound(CV_MAT_ELEM(*B, double, 0, 0) / CV_MAT_ELEM(*B, double, 2, 0));
		pt1.y = cvRound(CV_MAT_ELEM(*B, double, 1, 0) / CV_MAT_ELEM(*B, double, 2, 0));
		pt2.x = cvRound(CV_MAT_ELEM(*B, double, 0, 1) / CV_MAT_ELEM(*B, double, 2, 1));
		pt2.y = cvRound(CV_MAT_ELEM(*B, double, 1, 1) / CV_MAT_ELEM(*B, double, 2, 1));
		cvLine(inputImage, pt1, pt2, CV_RGB(255,0,0), 1, CV_AA, 0);
		pt1.x = cvRound(CV_MAT_ELEM(*B, double, 0, 1) / CV_MAT_ELEM(*B, double, 2, 1));
		pt1.y = cvRound(CV_MAT_ELEM(*B, double, 1, 1) / CV_MAT_ELEM(*B, double, 2, 1));
		pt2.x = cvRound(CV_MAT_ELEM(*B, double, 0, 2) / CV_MAT_ELEM(*B, double, 2, 2));
		pt2.y = cvRound(CV_MAT_ELEM(*B, double, 1, 2) / CV_MAT_ELEM(*B, double, 2, 2));
		cvLine(inputImage, pt1, pt2, CV_RGB(255,0,0), 1, CV_AA, 0);
		pt1.x = cvRound(CV_MAT_ELEM(*B, double, 0, 2) / CV_MAT_ELEM(*B, double, 2, 2));
		pt1.y = cvRound(CV_MAT_ELEM(*B, double, 1, 2) / CV_MAT_ELEM(*B, double, 2, 2));
		pt2.x = cvRound(CV_MAT_ELEM(*B, double, 0, 3) / CV_MAT_ELEM(*B, double, 2, 3));
		pt2.y = cvRound(CV_MAT_ELEM(*B, double, 1, 3) / CV_MAT_ELEM(*B, double, 2, 3));
		cvLine(inputImage, pt1, pt2, CV_RGB(255,0,0), 1, CV_AA, 0);
		pt1.x = cvRound(CV_MAT_ELEM(*B, double, 0, 3) / CV_MAT_ELEM(*B, double, 2, 3));
		pt1.y = cvRound(CV_MAT_ELEM(*B, double, 1, 3) / CV_MAT_ELEM(*B, double, 2, 3));
		pt2.x = cvRound(CV_MAT_ELEM(*B, double, 0, 0) / CV_MAT_ELEM(*B, double, 2, 0));
		pt2.y = cvRound(CV_MAT_ELEM(*B, double, 1, 0) / CV_MAT_ELEM(*B, double, 2, 0));
		cvLine(inputImage, pt1, pt2, CV_RGB(255,0,0), 1, CV_AA, 0);

		cvShowImage(inputWindowName, inputImage);
		cvShowImage(targetWindowName, targetImage);

		cvReleaseMat(&A);
		cvReleaseMat(&B);
	}

	//
	{
		CvMat *invH = cvCreateMat(3, 3, CV_64FC1);
		cvInvert(homography, invH, CV_LU);  // CV_LU or CV_SVD or CV_SVD_SYM

		const int cornerCount = 4;
		CvMat *A = cvCreateMat(3, cornerCount, CV_64FC1);
		CvMat *B = cvCreateMat(3, cornerCount, CV_64FC1);
		CV_MAT_ELEM(*A, double, 0, 0) = 0;
		CV_MAT_ELEM(*A, double, 1, 0) = 0;
		CV_MAT_ELEM(*A, double, 2, 0) = 1;
		CV_MAT_ELEM(*A, double, 0, 1) = inputImage->width - 1;
		CV_MAT_ELEM(*A, double, 1, 1) = 0;
		CV_MAT_ELEM(*A, double, 2, 1) = 1;
		CV_MAT_ELEM(*A, double, 0, 2) = inputImage->width - 1;
		CV_MAT_ELEM(*A, double, 1, 2) = inputImage->height - 1;
		CV_MAT_ELEM(*A, double, 2, 2) = 1;
		CV_MAT_ELEM(*A, double, 0, 3) = 0;
		CV_MAT_ELEM(*A, double, 1, 3) = inputImage->height - 1;
		CV_MAT_ELEM(*A, double, 2, 3) = 1;

		cvMatMul(invH, A, B);

		double minX = std::numeric_limits<double>::max(), maxX = -std::numeric_limits<double>::max(), minY = std::numeric_limits<double>::max(), maxY = -std::numeric_limits<double>::max();
		for (int i = 0; i < cornerCount; ++i)
		{
			const double x = CV_MAT_ELEM(*B, double, 0, i) / CV_MAT_ELEM(*B, double, 2, i);
			const double y = CV_MAT_ELEM(*B, double, 1, i) / CV_MAT_ELEM(*B, double, 2, i);

			if (x < minX) minX = x;
			else if (x > maxX) maxX = x;
			if (y < minY) minY = y;
			else if (y > maxY) maxY = y;
		}

		const int width = cvRound(maxX - minX), height = cvRound(maxY - minY);
		IplImage *image = cvCreateImage(cvSize(width, height), inputImage->depth, inputImage->nChannels);
		image->origin = image->origin;
		cvSetZero(image);
		const double scaleFactor = 640.0 / width;
		resultImage = cvCreateImage(cvSize(cvRound(width * scaleFactor), cvRound(height * scaleFactor)), inputImage->depth, inputImage->nChannels);
		resultImage->origin = inputImage->origin;
		cvSetZero(resultImage);

#if 0
		//cvWarpAffine(inputImage, image, homography, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP, cvScalarAll(0));
		//cvWarpAffine(inputImage, image, invH, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
	
		cvWarpPerspective(inputImage, image, homography, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP, cvScalarAll(0));
		//cvWarpPerspective(inputImage, image, invH, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
#elif 1
		CvPoint2D32f *srcPoints = (CvPoint2D32f *)cvAlloc(4 * sizeof(CvPoint2D32f));
		CvPoint2D32f *dstPoints = (CvPoint2D32f *)cvAlloc(4 * sizeof(CvPoint2D32f));
		CvMat *T = cvCreateMat(3, 3, CV_64FC1);

		srcPoints[0].x = 0;
		srcPoints[0].y = 0;
		srcPoints[1].x = (float)inputImage->width - 1;
		srcPoints[1].y = 0;
		srcPoints[2].x = (float)inputImage->width - 1;
		srcPoints[2].y = (float)inputImage->height - 1;
		srcPoints[3].x = 0;
		srcPoints[3].y = (float)inputImage->height - 1;
		dstPoints[0].x = (float)CV_MAT_ELEM(*B, double, 0, 0) / (float)CV_MAT_ELEM(*B, double, 2, 0) - (float)minX;
		dstPoints[0].y = (float)CV_MAT_ELEM(*B, double, 1, 0) / (float)CV_MAT_ELEM(*B, double, 2, 0) - (float)minY;
		dstPoints[1].x = (float)CV_MAT_ELEM(*B, double, 0, 1) / (float)CV_MAT_ELEM(*B, double, 2, 1) - (float)minX;
		dstPoints[1].y = (float)CV_MAT_ELEM(*B, double, 1, 1) / (float)CV_MAT_ELEM(*B, double, 2, 1) - (float)minY;
		dstPoints[2].x = (float)CV_MAT_ELEM(*B, double, 0, 2) / (float)CV_MAT_ELEM(*B, double, 2, 2) - (float)minX;
		dstPoints[2].y = (float)CV_MAT_ELEM(*B, double, 1, 2) / (float)CV_MAT_ELEM(*B, double, 2, 2) - (float)minY;
		dstPoints[3].x = (float)CV_MAT_ELEM(*B, double, 0, 3) / (float)CV_MAT_ELEM(*B, double, 2, 3) - (float)minX;
		dstPoints[3].y = (float)CV_MAT_ELEM(*B, double, 1, 3) / (float)CV_MAT_ELEM(*B, double, 2, 3) - (float)minY;

		cvGetPerspectiveTransform(srcPoints, dstPoints, T);

		//for (int i = 0; i < 3; ++i)
		//{
		//	for (int j = 0; j < 3; ++j)
		//		std::cout << CV_MAT_ELEM(*homography, double, i, j) << ", ";
		//	std::cout << std::endl;
		//}
		//std::cout << std::endl;
		//for (int i = 0; i < 3; ++i)
		//{
		//	for (int j = 0; j < 3; ++j)
		//		std::cout << CV_MAT_ELEM(*invH, double, i, j) << ", ";
		//	std::cout << std::endl;
		//}
		//std::cout << std::endl;
		//for (int i = 0; i < 3; ++i)
		//{
		//	for (int j = 0; j < 3; ++j)
		//		std::cout << CV_MAT_ELEM(*T, double, i, j) << ", ";
		//	std::cout << std::endl;
		//}
		//std::cout << std::endl;

		cvWarpPerspective(inputImage, image, T, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

		cvReleaseMat(&T);
		cvFree(&srcPoints);
		cvFree(&dstPoints);
#else
		CvMat *vi = cvCreateMat(3, 1, CV_64FC1);
		CvMat *vo = cvCreateMat(3, 1, CV_64FC1);

		for (int row = 0; row < inputImage->height; ++row)
			for (int col = 0; col < inputImage->width; ++col)
			{
				CV_MAT_ELEM(*vi, double, 0, 0) = col;
				CV_MAT_ELEM(*vi, double, 1, 0) = row;
				CV_MAT_ELEM(*vi, double, 2, 0) = 1;

				cvMatMul(invH, vi, vo);

				const int x = cvRound(CV_MAT_ELEM(*vo, double, 0, 0) / CV_MAT_ELEM(*vo, double, 2, 0) - minX);
				const int y = cvRound(CV_MAT_ELEM(*vo, double, 1, 0) / CV_MAT_ELEM(*vo, double, 2, 0) - minY);

				if (x < width && y < height)
				{
					CV_IMAGE_ELEM(image, unsigned char, y, x * image->nChannels) = CV_IMAGE_ELEM(inputImage, unsigned char, row, col * inputImage->nChannels);
					CV_IMAGE_ELEM(image, unsigned char, y, x * image->nChannels + 1) = CV_IMAGE_ELEM(inputImage, unsigned char, row, col * inputImage->nChannels + 1);
					CV_IMAGE_ELEM(image, unsigned char, y, x * image->nChannels + 2) = CV_IMAGE_ELEM(inputImage, unsigned char, row, col * inputImage->nChannels + 2);
				}
				else
					std::cout << x << ", " << y << std::endl;
			}

		cvReleaseMat(&vi);
		cvReleaseMat(&vo);
#endif

		const int interpolation = CV_INTER_LINEAR;  // CV_INTER_NN, CV_INTER_LINEAR, CV_INTER_AREA, CV_INTER_CUBIC
		cvResize(image, resultImage, interpolation);

		cvReleaseMat(&A);
		cvReleaseMat(&B);

		cvReleaseImage(&image);
		cvReleaseMat(&invH);

		cvShowImage(resultantWindowName, resultImage);
	}

	cvWaitKey();

	//
	cvReleaseMat(&homography);

	cvReleaseImage(&targetImage);
	cvReleaseImage(&inputImage);
	cvReleaseImage(&resultImage);

	cvClearMemStorage(targetStorage);
	cvClearMemStorage(inputStorage);

	cvDestroyWindow(targetWindowName);
	cvDestroyWindow(inputWindowName);
	cvDestroyWindow(resultantWindowName);
}
