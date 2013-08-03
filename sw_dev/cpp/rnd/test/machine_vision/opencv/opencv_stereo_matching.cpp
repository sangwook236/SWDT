//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void stereo_correspondence_using_block_matching_algorithm__c_version(const cv::Mat &leftImage, const cv::Mat &rightImage, const int numberOfDisparities, cv::Mat &disparity)
{
	const int preset = CV_STEREO_BM_BASIC;
	const int SADWindowSize = 9;
	const int speckleWindowSize = 100;
	const int speckleRange = 32;

	CvStereoBMState *state = cvCreateStereoBMState(preset, numberOfDisparities);
	state->SADWindowSize = SADWindowSize;
	state->speckleWindowSize = speckleWindowSize;
	state->speckleRange = speckleRange;

	// computes the disparity map using block matching algorithm
	disparity = cv::Mat::zeros(leftImage.size(), CV_32FC1);  // disparity: single-channel 16-bit signed, or 32-bit floating-point disparity map
#if defined(__GNUC__)
    {
        IplImage leftImage_ipl = (IplImage)leftImage;
        IplImage rightImage_ipl = (IplImage)rightImage;
        IplImage disparity_ipl = (IplImage)disparity;
        cvFindStereoCorrespondenceBM(&leftImage_ipl, &rightImage_ipl, &disparity_ipl, state);
    }
#else
	cvFindStereoCorrespondenceBM(&(IplImage)leftImage, &(IplImage)rightImage, &(IplImage)disparity, state);
#endif

	cvReleaseStereoBMState(&state);
}

void stereo_correspondence_using_graph_cut_based_algorithm__c_version(const cv::Mat &leftImage, const cv::Mat &rightImage, const int numberOfDisparities, cv::Mat &leftDisparity, cv::Mat &rightDisparity)
{
	const int maxIters = 100;  // ???
	CvStereoGCState *state = cvCreateStereoGCState(numberOfDisparities, maxIters);

	// computes the disparity map using graph cut-based algorithm
	const int useDisparityGuess = 0;
	leftDisparity = cv::Mat::zeros(leftImage.size(), CV_16SC1);  // disparity: single-channel 16-bit signed left disparity map
	rightDisparity = cv::Mat::zeros(leftImage.size(), CV_16SC1);  // disparity: single-channel 16-bit signed left disparity map
#if defined(__GNUC__)
    {
        IplImage leftImage_ipl = (IplImage)leftImage;
        IplImage rightImage_ipl = (IplImage)rightImage;
        IplImage leftDisparity_ipl = (IplImage)leftDisparity;
        IplImage rightDisparity_ipl = (IplImage)rightDisparity;
        cvFindStereoCorrespondenceGC(&leftImage_ipl, &rightImage_ipl, &leftDisparity_ipl, &rightDisparity_ipl, state, useDisparityGuess);
    }
#else
	cvFindStereoCorrespondenceGC(&(IplImage)leftImage, &(IplImage)rightImage, &(IplImage)leftDisparity, &(IplImage)rightDisparity, state, useDisparityGuess);
#endif

	cvReleaseStereoGCState(&state);
}

enum SimilarityComparisonMethod
{
	SCM_SAD,  // sum of absolute differences
	SCM_LSAD,  // locally scaled sum of absolute differences
	SCM_ZSAD,  // zero-mean sum of absolute differences
	SCM_SSD,  // sum of squared differences
	SCM_LSSD,  // locally scaled sum of squared differences
	SCM_ZSSD,  // zero-mean sum of squared differences
	SCM_NCC,  // normalized cross correlation
	SCM_ZNCC,  // zero-mean normalized cross correlation
	SCM_SHD  // sum of Hamming distances
};

#define __USE_LEFT_IMAGE_AS_REFERENCE 1

void stereo_correspondence_using_similarity_measure(const cv::Mat &leftImage, const cv::Mat &rightImage, const int windowSize, const int minDisparity, const int maxDisparity, const SimilarityComparisonMethod &method, cv::Mat &disparity)
// [ref] http://siddhantahuja.wordpress.com/tag/sum-of-squared-differences/
// leftImage
// rightImage
// windowSize: window size
// minDisparity: minimum disparity in x-direction
// maxDisparity: maximum disparity in x-direction
// method: method used for calculating the similarity measure
{
	if (leftImage.rows != rightImage.rows || leftImage.cols != rightImage.cols)
	{
		std::cout << "the size of two images is different" << std::endl;
		return;
	}
	if (windowSize % 2 == 0)
	{
		std::cout << "the window size must be an odd number" << std::endl;
		return;
	}
	if (minDisparity > maxDisparity)
	{
		std::cout << "the minimum disparity must be less than the maximum disparity" << std::endl;
		return;
	}

	const int rows = leftImage.rows;
	const int cols = leftImage.cols;

	const int win = (windowSize - 1) / 2;
	const bool isMaximized = SCM_NCC == method || SCM_ZNCC == method;

	cv::Mat img_ref_f, img_target_f;
#if __USE_LEFT_IMAGE_AS_REFERENCE
	// reference image: left image
	leftImage.convertTo(img_ref_f, CV_32FC1, 1.0, 0.0);
	rightImage.convertTo(img_target_f, CV_32FC1, 1.0, 0.0);
#else
	// reference right: right image
	rightImage.convertTo(img_ref_f, CV_32FC1, 1.0, 0.0);
	leftImage.convertTo(img_target_f, CV_32FC1, 1.0, 0.0);
#endif

#if 0
	cv::Mat meanMat_ref(cv::Mat::zeros(rows, cols, CV_32FC1)), meanMat_target(cv::Mat::zeros(rows, cols, CV_32FC1));
	for (int r = win; r < rows - win; ++r)
	{
		for (int c = win; c < cols - win; ++c)
		{
			//meanMat_ref.at<float>(r, c) = (float)cv::mean(img_ref_f(cv::Rect(c-win, r-win, windowSize, windowSize)))[0];
			//meanMat_target.at<float>(r, c) = (float)cv::mean(img_target_f(cv::Rect(c-win, r-win, windowSize, windowSize)))[0];
			meanMat_ref.at<float>(r, c) = (float)cv::mean(img_ref_f(cv::Range(r-win, r+win+1), cv::Range(c-win, c+win+1)))[0];
			meanMat_target.at<float>(r, c) = (float)cv::mean(img_target_f(cv::Range(r-win, r+win+1), cv::Range(c-win, c+win+1)))[0];
		}
	}
#else
	// mean filter
	cv::Mat meanMat_ref(cv::Mat::zeros(rows, cols, CV_32FC1)), meanMat_target(cv::Mat::zeros(rows, cols, CV_32FC1));
	{
		cv::boxFilter(img_ref_f, meanMat_ref, -1, cv::Size(windowSize, windowSize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
		cv::boxFilter(img_target_f, meanMat_target, -1, cv::Size(windowSize, windowSize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
	}
#endif

	//
	disparity = cv::Mat::zeros(rows, cols, CV_16SC1) * (minDisparity - 1);  // disparity: 16-bit signed single-channel image
	const double eps = 1.0e-20;
	float score, weight, mean_ref, mean_target;
	for (int r = win; r < rows - win; ++r)
	{
		for (int c = win; c < cols - win; ++c)
		{
#if __USE_LEFT_IMAGE_AS_REFERENCE
			// reference image: left image
			if (c - minDisparity < win || c - maxDisparity >= cols - win) continue;

			//const cv::Mat roi_ref(img_ref_f, cv::Rect(c-win, r-win, windowSize, windowSize));
			const cv::Mat roi_ref(img_ref_f, cv::Range(r-win, r+win+1), cv::Range(c-win, c+win+1));
			mean_ref = meanMat_ref.at<float>(r, c);
#else
			// reference image: right image
			if (c + maxDisparity < win || c + minDisparity >= cols - win) continue;

			//const cv::Mat roi_ref(img_target_f, cv::Rect(c-win, r-win, windowSize, windowSize));
			const cv::Mat roi_ref(img_ref_f, cv::Range(r-win, r+win+1), cv::Range(c-win, c+win+1));
			mean_ref = meanMat_ref.at<float>(r, c);
#endif

			double bestMatchScore = isMaximized ? 0.0 : std::numeric_limits<double>::max();
			short bestMatchIdx = minDisparity - 1;
			for (int d = minDisparity; d <= maxDisparity; ++d)
			{
#if __USE_LEFT_IMAGE_AS_REFERENCE
				// reference image: left image
				if (c - d < win || c - d >= cols - win) continue;

				//const cv::Mat roi_target(img_target_f, cv::Rect(c-d-win, r-win, windowSize, windowSize));
				const cv::Mat roi_target(img_target_f, cv::Range(r-win, r+win+1), cv::Range(c-d-win, c-d+win+1));
				mean_target = meanMat_target.at<float>(r, c - d);
#else
				// reference image: right image
				if (c + d < win || c + d >= cols - win) continue;

				//const cv::Mat roi_target(img_ref_f, cv::Rect(c+d-win, r-win, windowSize, windowSize));
				const cv::Mat roi_target(img_ref_f, cv::Range(r-win, r+win+1), cv::Range(c+d-win, c+d+win+1));
				mean_target = meanMat_target.at<float>(r, c + d);
#endif

				switch (method)
				{
				case SCM_SAD:  // sum of absolute differences
					score = (float)cv::norm(roi_ref, roi_target, cv::NORM_L1);
					break;
				case SCM_LSAD:  // locally scaled sum of absolute differences
					weight = std::fabs(mean_target) <= eps ? 0.0f : (mean_ref / mean_target);
					score = (float)cv::norm(roi_ref, roi_target * weight, cv::NORM_L1);
					break;
				case SCM_ZSAD:  // zero-mean sum of absolute differences
					score = (float)cv::norm(roi_ref - mean_ref, roi_target - mean_target, cv::NORM_L1);
					break;
				case SCM_SSD:  // sum of squared differences
					score = (float)cv::norm(roi_ref, roi_target, cv::NORM_L2);
					break;
				case SCM_LSSD:  // locally scaled sum of squared differences
					weight = std::fabs(mean_target) <= eps ? 0.0f : (mean_ref / mean_target);
					score = (float)cv::norm(roi_ref, roi_target * weight, cv::NORM_L2);
					break;
				case SCM_ZSSD:  // zero-mean sum of squared differences
					score = (float)cv::norm(roi_ref - mean_ref, roi_target - mean_target, cv::NORM_L2);
					break;
				case SCM_NCC:  // normalized cross correlation
					weight = (float)cv::norm(roi_ref, cv::NORM_L2) * (float)cv::norm(roi_target, cv::NORM_L2);
					score = (float)cv::sum(roi_ref.mul(roi_target, 1.0 / weight))[0];
					break;
				case SCM_ZNCC:  // zero-mean normalized cross correlation
					{
						const cv::Mat m1(roi_ref - mean_ref), m2(roi_target - mean_target);
						weight = (float)cv::norm(m1, cv::NORM_L2) * (float)cv::norm(m2, cv::NORM_L2);
						score = (float)cv::sum(m1.mul(m2, 1.0 / weight))[0];
					}
					break;
				case SCM_SHD:  // sum of Hamming distances
#if 0
					score = (float)cv::sum(roi_ref ^ roi_target)[0];
#else
#if __USE_LEFT_IMAGE_AS_REFERENCE
					// reference image: left image
					score = (float)cv::sum(leftImage(cv::Range(r-win, r+win+1), cv::Range(c-win, c+win+1)) ^ rightImage(cv::Range(r-win, r+win+1), cv::Range(c-d-win, c-d+win+1)))[0];
#else
					// reference image: right image
					score = (float)cv::sum(rightImage(cv::Range(r-win, r+win+1), cv::Range(c-win, c+win+1)) ^ leftImage(cv::Range(r-win, r+win+1), cv::Range(c+d-win, c+d+win+1)))[0];
#endif
#endif
					break;
				default:
					throw std::runtime_error("undefined comparison method");
				}

				if (isMaximized)
				{
					if (score > bestMatchScore)
					{
						bestMatchScore = score;
						bestMatchIdx = (short)d;
					}
				}
				else
				{
					if (score < bestMatchScore)
					{
						bestMatchScore = score;
						bestMatchIdx = (short)d;
					}
				}
			}

			disparity.at<short>(r, c) = bestMatchIdx;
		}
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// [ref] ${OPENCV_HOME}/samples/cpp/stereo_match.cpp

void stereo_matching()
{
	const std::string filename1("./data/machine_vision/opencv/scene_l.bmp");
	const std::string filename2("./data/machine_vision/opencv/scene_r.bmp");

	const cv::Mat &left_image = cv::imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
	const cv::Mat &right_image = cv::imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);

	//
	cv::Mat disparity8;

	const double t = (double)cv::getTickCount();
	{
		// rectify stereo image pair.

		// stereo matching.
#if 0
		// METHOD #1: using block matching algorithm (C version).

		cv::Mat disparity;

		//const int num_disparities = left_image.cols / 8;
		const int num_disparities = 16;
		local::stereo_correspondence_using_block_matching_algorithm__c_version(left_image, right_image, num_disparities, disparity);

		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(disparity, &minVal, &maxVal);
		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		disparity.convertTo(disparity8, CV_8U, alpha, beta);
#elif 0
		// METHOD #2: using graph cut-based algorithm (C version).

		cv::Mat left_disparity, right_disparity;

		//const int num_disparities = 20;
		const int num_disparities = 16;
		local::stereo_correspondence_using_graph_cut_based_algorithm__c_version(left_image, right_image, num_disparities, left_disparity, right_disparity);

		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(left_disparity, &minVal, &maxVal);
		//cv::minMaxLoc(right_disparity, &minVal, &maxVal);
		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		left_disparity.convertTo(disparity8, CV_8U, alpha, beta);
		//right_disparity.convertTo(disparity8, CV_8U, alpha, beta);
#elif 0
		// METHOD #3: using block matching algorithm (C++ version).

		cv::Mat disparity;

		//const int num_disparities = (left_image.cols / 8 + 15) & -16;
		const int num_disparities = 16;

		const int SADWindowSize = 9;
		const int speckleWindowSize = 100;
		const int speckleRange = 32;

		// computing stereo correspondence using block matching algorithm.
		cv::StereoBM bm;
		bm.state->roi1 = cvRect(0, 0, left_image.cols, left_image.rows);
		bm.state->roi2 = cvRect(0, 0, right_image.cols, right_image.rows);
		bm.state->preFilterCap = 31;
		bm.state->SADWindowSize = SADWindowSize;
		bm.state->minDisparity = 0;
		bm.state->numberOfDisparities = num_disparities;  // maximum disparity - minimum disparity (> 0).
		bm.state->textureThreshold = 10;
		bm.state->uniquenessRatio = 15;
		bm.state->speckleWindowSize = speckleWindowSize;
		bm.state->speckleRange = speckleRange;
		bm.state->disp12MaxDiff = 1;

		bm(left_image, right_image, disparity);  // disparity: 16SC1.

		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(disparity, &minVal, &maxVal);
		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		disparity.convertTo(disparity8, CV_8U, alpha, beta);
#elif 0
		// METHOD #4: using semi-global block matching algorithm (C++ version).

		cv::Mat disparity;

		//const int num_disparities = (left_image.cols / 8 + 15) & -16;
		const int num_disparities = 16;

		const int SADWindowSize = 3;
		const int speckleWindowSize = 100;
		const int speckleRange = 32;

		const int channels = left_image.channels();

		// computing stereo correspondence using semi-global block matching algorithm.
		cv::StereoSGBM sgbm;
		sgbm.preFilterCap = 63;
		sgbm.SADWindowSize = SADWindowSize;
		sgbm.P1 = 8 * channels * sgbm.SADWindowSize * sgbm.SADWindowSize;
		sgbm.P2 = 32 * channels * sgbm.SADWindowSize * sgbm.SADWindowSize;
		sgbm.minDisparity = 0;
		sgbm.numberOfDisparities = num_disparities;  // maximum disparity - minimum disparity (> 0).
		sgbm.uniquenessRatio = 10;
		sgbm.speckleWindowSize = speckleWindowSize;
		sgbm.speckleRange = speckleRange;
		sgbm.disp12MaxDiff = 1;
		// if false, a single-pass, which means that you consider only 5 directions instead of 8.
		// if true, the full-scale two-pass dynamic programming algorithm.
		sgbm.fullDP = true;

		sgbm(left_image, right_image, disparity);  // disparity: 16SC1.

		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(disparity, &minVal, &maxVal);
		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		disparity.convertTo(disparity8, CV_8U, alpha, beta);
#elif 0
		// METHOD #5: using the variational matching algorithm (C++ version).

		cv::Mat disparity;

		//const int num_disparities = (left_image.cols / 8 + 15) & -16;
		const int num_disparities = 16;

		// computing stereo correspondence using the variational matching algorithm.
		cv::StereoVar var;
		var.levels = 3;  // ignored if flag USE_AUTO_PARAMS is set.
		var.pyrScale = 0.5;  // ignored if flag USE_AUTO_PARAMS is set.
		var.nIt = 25;
		var.minDisp = -num_disparities;
		var.maxDisp = 0;
		var.poly_n = 3;
		var.poly_sigma = 0.0;
		var.fi = 15.0f;
		var.lambda = 0.03f;
		var.penalization = var.PENALIZATION_TICHONOV;  // ignored if flag USE_AUTO_PARAMS is set.
		var.cycle = var.CYCLE_V;  // ignored if flag USE_AUTO_PARAMS is set.
		var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;

		var(left_image, right_image, disparity);

		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(disparity, &minVal, &maxVal);
		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		disparity.convertTo(disparity8, CV_8U, alpha, beta);
#elif 1
		// METHOD #6: using similarity measure-based matching algorithm (C++ version).

		cv::Mat disparity;

		const int window_size = 9;
		const int min_disparity = 0;
		const int max_disparity = 16;
		// SAD, LSAD, ZSAD, SSD, LSSD, ZSSD, NCC, ZNCC, SHD
		const local::SimilarityComparisonMethod method = local::SCM_SAD;
		local::stereo_correspondence_using_similarity_measure(left_image, right_image, window_size, min_disparity, max_disparity, method, disparity);
		//local::stereo_correspondence_using_similarity_measure(right_image, left_image, window_size, min_disparity, max_disparity, method, disparity);

		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(disparity, &minVal, &maxVal);
		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		disparity.convertTo(disparity8, CV_8U, alpha, beta);
#endif
	}
	const double et = ((double)cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();
	std::cout << "time elapsed: " << et << "ms" << std::endl;

	//
	cv::imshow("stereo matching - left image", left_image);
	cv::imshow("stereo matching - right image", right_image);
	cv::imshow("stereo matching - disparity", disparity8);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

}  // namespace my_opencv
