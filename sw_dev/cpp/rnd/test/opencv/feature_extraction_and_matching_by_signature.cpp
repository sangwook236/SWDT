#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <vector>


namespace {
namespace local {

void feature_extraction_and_matching_by_signature_1()
{
#if 1
	const std::string img1_name("opencv_data\\box.png");
	const std::string img2_name("opencv_data\\box_in_scene.png");
#elif 0
	const std::string img1_name("opencv_data\\melon_target.png");
	const std::string img2_name("opencv_data\\melon_1.png");
	//const std::string img2_name("opencv_data\\melon_2.png");
	//const std::string img2_name("opencv_data\\melon_3.png");
#endif

	//std::cout << "reading the images..." << std::endl;
	const cv::Mat &img1 = cv::imread(img1_name, CV_LOAD_IMAGE_GRAYSCALE);
	const cv::Mat &img2 = cv::imread(img2_name, CV_LOAD_IMAGE_GRAYSCALE);
	if (img1.empty() || img2.empty())
	{
		std::cout << "fail to load image files" << std::endl;
		return;
	}

	CvSeq *objectKeypoints = NULL, *objectDescriptors = NULL;
	CvSeq *imageKeypoints = NULL, *imageDescriptors = NULL;
	const CvSURFParams params = cvSURFParams(500, 1);
	CvMemStorage *storage = cvCreateMemStorage(0);

	cvExtractSURF(&(IplImage)img1, 0, &objectKeypoints, &objectDescriptors, storage, params);
	cvExtractSURF(&(IplImage)img2, 0, &imageKeypoints, &imageDescriptors, storage, params);

	cv::RTreeClassifier detector;
	const int patch_width = cv::RandomizedTree::PATCH_SIZE;
	const int patch_height = cv::RandomizedTree::PATCH_SIZE;
	std::vector<cv::BaseKeypoint> base_set;

	const int n_points = std::min(200, objectKeypoints->total);
	base_set.reserve(n_points);
	for (int i = 0; i < n_points; ++i)
	{
		CvSURFPoint *point = (CvSURFPoint *)cvGetSeqElem(objectKeypoints, i);
		base_set.push_back(cv::BaseKeypoint(point->pt.x, point->pt.y, &(IplImage)img1));
	}

	// train detector
	cv::RNG rng(cv::getTickCount());
	cv::PatchGenerator gen(0, 255, 2, false, 0.7, 1.3, -CV_PI/3, CV_PI/3, -CV_PI/3, CV_PI/3);
	std::cout << "RTree Classifier training..." << std::endl;
	detector.train(base_set, rng, gen, 24, cv::RandomizedTree::DEFAULT_DEPTH, 2000, (int)base_set.size(), detector.DEFAULT_NUM_QUANT_BITS);
	std::cout << "Done" << std::endl;

	float *best_corr = NULL;
	int *best_corr_idx = NULL;
	if (imageKeypoints->total > 0)
	{
		best_corr = new float [imageKeypoints->total];
		best_corr_idx = new int [imageKeypoints->total];
	}

	float *signature = new float [detector.original_num_classes()];
	for (int i = 0; i < imageKeypoints->total; ++i)
	{
		CvSURFPoint *point = (CvSURFPoint *)cvGetSeqElem(imageKeypoints, i);
		int part_idx = -1;
		float prob = 0.0f;
		CvRect roi = cvRect((int)(point->pt.x) - patch_width/2, (int)(point->pt.y) - patch_height/2, patch_width, patch_height);
		cvSetImageROI(&(IplImage)img2, roi);
		roi = cvGetImageROI(&(IplImage)img2);
		if(roi.width != patch_width || roi.height != patch_height)
		{
			best_corr_idx[i] = part_idx;
			best_corr[i] = prob;
		}
		else
		{
			cvSetImageROI(&(IplImage)img2, roi);

			IplImage *roi_image = cvCreateImage(cvSize(roi.width, roi.height), img2.depth(), img2.channels());
			cvCopy(&(IplImage)img2, roi_image);
			detector.getSignature(roi_image, signature);
			//detector.getSparseSignature(roi_image, signature, thres);

			for (int j = 0; j< detector.original_num_classes(); ++j)
			{
				if (prob < signature[j])
				{
					part_idx = j;
					prob = signature[j];
				}
			}
			best_corr_idx[i] = part_idx;
			best_corr[i] = prob;

			if (roi_image) cvReleaseImage(&roi_image);
		}

		cvResetImageROI(&(IplImage)img2);
	}

	delete [] best_corr;
	delete [] best_corr_idx;
	cvReleaseMemStorage(&storage);
}

}  // namespace local
}  // unnamed namespace

void feature_extraction_and_matching_by_signature()
{
	local::feature_extraction_and_matching_by_signature_1();
}
