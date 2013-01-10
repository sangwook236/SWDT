//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <string>
#include <iostream>


namespace my_opencv {

void print_opencv_matrix(const cv::Mat &mat);

}  // namespace my_opencv

namespace {
namespace local {

void image_handling()
{
	// element's type of an image
	std::cout << ">>> element's type of an image" << std::endl;
	{
		const int rdim = 10, cdim = 10;

		//const int iplDepth = IPL_DEPTH_8U;
		//const int iplDepth = IPL_DEPTH_8S;
		//const int iplDepth = IPL_DEPTH_16U;
		//const int iplDepth = IPL_DEPTH_16S;
		//const int iplDepth = IPL_DEPTH_32S;
		const int iplDepth = IPL_DEPTH_32F;
		//const int iplDepth = IPL_DEPTH_64F;

		IplImage *img = cvCreateImage(cvSize(rdim, cdim), iplDepth, 1);

		CV_IS_IMAGE_HDR(img);
		CV_IS_IMAGE(img);

		if (iplDepth == IPL_DEPTH_8U)
		{
			for (int i = 0, ii = 1; i < rdim; ++i)
				for (int j = 0; j < cdim; ++j, ++ii)
					//img->imageData[i*cdim + j] = (unsigned char)ii;
					CV_IMAGE_ELEM(img, unsigned char, i, j) = (unsigned char)ii;

			std::cout << "image =>" << std::endl;
			my_opencv::print_opencv_matrix(cv::Mat(rdim, cdim, CV_8UC1, (void *)img->imageData));
			//const unsigned char *p = (unsigned char *)img->imageData;
			//my_opencv::print_opencv_matrix(cv::Mat(rdim, cdim, CV_8UC1, (void*)p));
		}
		else if (iplDepth == IPL_DEPTH_32F)
		{
			float *p = (float *)img->imageData;
			for (int i = 0, ii = 1; i < rdim; ++i)
				for (int j = 0; j < cdim; ++j, ++ii)
					//p[i*cdim + j] = (float)ii;
					CV_IMAGE_ELEM(img, float, i, j) = (float)ii;

			std::cout << "image =>" << std::endl;
			my_opencv::print_opencv_matrix(cv::Mat(rdim, cdim, CV_32FC1, (void *)img->imageData));
			//p = (float *)img->imageData;
			//my_opencv::print_opencv_matrix(cv::Mat(rdim, cdim, CV_32FC1, (void *)p));
		}
		else
			assert(false);

		cvReleaseImage(&img);
	}
}

void show_image()
{
	const char *winName = "machine_vision_data\\opencv\\Image:";
	const char *imgName = "machine_vision_data\\opencv\\lena_gray.bmp";
	const char *savedImgName = "machine_vision_data\\opencv\\lena_gray_edge.png";

	IplImage *img = cvLoadImage(imgName);
	IplImage *grayImg = 0L;
	if (1 == img->nChannels)
		grayImg = img;
	else
	{
		grayImg = cvCreateImage(cvGetSize(img), img->depth, 1);
#if defined(__GNUC__)
		if (strcasecmp(img->channelSeq, "RGB") == 0)
#elif defined(_MSC_VER)
		if (_stricmp(img->channelSeq, "RGB") == 0)
#endif
			cvCvtColor(img, grayImg, CV_RGB2GRAY);
#if defined(__GNUC__)
		else if (strcasecmp(img->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
		else if (_stricmp(img->channelSeq, "BGR") == 0)
#endif
			cvCvtColor(img, grayImg, CV_BGR2GRAY);
		else
			assert(false);
	}
	assert(grayImg);
	IplImage *edgeImg = cvCreateImage(cvGetSize(grayImg), grayImg->depth, grayImg->nChannels);
	//IplImage *edgeImg = cvCloneImage(grayImg);

	//cvNamedWindow(winName, CV_WINDOW_AUTOSIZE);
	//cvShowImage(winName, grayImg);

	cvCanny(grayImg, edgeImg, 30.0, 90.0, 3);
	cvNamedWindow(winName, CV_WINDOW_AUTOSIZE);
	cvShowImage(winName, edgeImg);

	cvSaveImage(savedImgName, edgeImg);

	cvWaitKey();
	cvDestroyWindow(winName);
    cvReleaseImage(&img);
    cvReleaseImage(&grayImg);
    cvReleaseImage(&edgeImg);
}

void canny(const cv::Mat &gray, cv::Mat &edge)
{
#if 0
	// down-scale and up-scale the image to filter out the noise
	cv::Mat blurred;
	cv::pyrDown(gray, blurred);
	cv::pyrUp(blurred, edge);
#else
	cv::blur(gray, edge, cv::Size(3, 3));
#endif

	// run the edge detector on grayscale
	const int lowerEdgeThreshold = 30, upperEdgeThreshold = 50;
	const bool useL2 = true;
	cv::Canny(edge, edge, lowerEdgeThreshold, upperEdgeThreshold, 3, useL2);
}

void sobel(const cv::Mat &gray, cv::Mat &edge)
{
	//const int ksize = 5;
	const int ksize = CV_SCHARR;
	cv::Mat xgradient, ygradient;

	cv::Sobel(gray, xgradient, CV_32FC1, 1, 0, ksize, 1.0, 0.0);
	cv::Sobel(gray, ygradient, CV_32FC1, 0, 1, ksize, 1.0, 0.0);

	cv::magnitude(xgradient, ygradient, edge);
}

void matches2points(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::Point2f> &matchPoints1, std::vector<cv::Point2f> &matchPoints2)
{
	const size_t &count = matches.size();
	matchPoints1.clear();
	matchPoints2.clear();
	matchPoints1.reserve(count);
	matchPoints2.reserve(count);
	for (size_t i = 0; i < count; ++i)
	{
		const cv::DMatch &match = matches[i];
		//matchPoints2.push_back(keypoints2[match.queryIdx].pt);
		//matchPoints1.push_back(keypoints1[match.trainIdx].pt);
		matchPoints2.push_back(keypoints2[match.trainIdx].pt);
		matchPoints1.push_back(keypoints1[match.queryIdx].pt);
	}
}

void image_subtraction()
{
	const std::string img1_name("machine_vision_data\\opencv\\table_only.jpg");
	const std::string img2_name("machine_vision_data\\opencv\\table_hand_01.jpg");
	//const std::string img2_name("machine_vision_data\\opencv\\table_hand_02.jpg");
	//const std::string img2_name("machine_vision_data\\opencv\\table_hand_03.jpg");
	//const std::string img2_name("machine_vision_data\\opencv\\table_hand_04.jpg");
	//const std::string img2_name("machine_vision_data\\opencv\\table_hand_05.jpg");

	const cv::Mat &img1 = cv::imread(img1_name, CV_LOAD_IMAGE_GRAYSCALE);
	const cv::Mat &img2 = cv::imread(img2_name, CV_LOAD_IMAGE_GRAYSCALE);
	if (img1.empty() || img2.empty())
	{
		std::cout << "fail to load image files" << std::endl;
		return;
	}

	cv::Mat processed_img1, processed_img2;
#if 0
	canny(img1, processed_img1);
	canny(img2, processed_img2);
#elif 0
	sobel(img1, processed_img1);
	sobel(img2, processed_img2);
#else
	processed_img1 = img1;
	processed_img2 = img2;
#endif

	//processed_img1 = processed_img1 > 0;
	//processed_img2 = processed_img2 > 0;

/*
	{
		std::vector<cv::KeyPoint> keypoints1, keypoints2;
		cv::SiftFeatureDetector featureDetector;
		featureDetector.detect(img1, keypoints1);
		featureDetector.detect(img2, keypoints2);

		cv::Mat descriptors1, descriptors2;
		cv::SiftDescriptorExtractor extractor;
		extractor.compute(img1, keypoints1, descriptors1);
		extractor.compute(img2, keypoints2, descriptors2);

		//cv::BruteForceMatcher<cv::L1<float> > descriptorMatcher;
		cv::BruteForceMatcher<cv::L2<float> > descriptorMatcher;
		//cv::FlannBasedMatcher descriptorMatcher;
		std::vector<cv::DMatch> matches;
		descriptorMatcher.match(descriptors1, descriptors2, matches);

		if (!matches.empty())
		{
			std::vector<cv::Point2f> matchedPoints1, matchedPoints2;
			matches2points(matches, keypoints1, keypoints2, matchedPoints1, matchedPoints2);  // extract a list of the (x,y) location of the matches
			std::vector<unsigned char> outlier_mask;
			const double ransacReprojThreshold = 3.0;
			const cv::Mat &H = cv::findHomography(cv::Mat(matchedPoints1), cv::Mat(matchedPoints2), outlier_mask, cv::RANSAC, ransacReprojThreshold);
			//const cv::Mat &H = cv::findHomography(cv::Mat(matchedPoints1), cv::Mat(matchedPoints2), outlier_mask, cv::LMEDS, ransacReprojThreshold);

			for (size_t i = 0; i < H.rows; ++i)
			{
				for (size_t j = 0; j < H.cols; ++j)
				{
					// caution !!! data type is double, but not float
					//std::cout << H.at<float>(i,j) << ", ";
					std::cout << H.at<double>(i,j) << ", ";
				}
				std::cout << std::endl;
			}

			cv::Mat img_correspondence;
			cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_correspondence, cv::Scalar::all(-1), cv::Scalar::all(-1), reinterpret_cast<const std::vector<char> &>(outlier_mask));
			cv::Mat img_warped;
			cv::warpPerspective(img1, img_warped, H, img2.size());

			const std::string winName1("image subtraction - correspondence");
			const std::string winName2("image subtraction - warped");
			cv::namedWindow(winName1, cv::WINDOW_AUTOSIZE);
			cv::namedWindow(winName2, cv::WINDOW_AUTOSIZE);

			cv::imshow(winName1, img_correspondence);
			cv::imshow(winName2, img_warped);
		}
	}
*/

	cv::Mat processed_diff_img;
	cv::absdiff(processed_img1, processed_img2, processed_diff_img);
	processed_diff_img = processed_diff_img > 64;

	cv::Mat filtered_img;
	processed_img2.copyTo(filtered_img, processed_diff_img);

	//
	const std::string windowName1("image subtraction - bg only");
	const std::string windowName2("image subtraction - fg + bg");
	const std::string windowName3("image subtraction - result");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName3, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName1, processed_img1);
	cv::imshow(windowName2, processed_img2);
	cv::imshow(windowName3, filtered_img);

	const unsigned char key = cv::waitKey(0);
	if (27 == key)
		return;

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
	cv::destroyWindow(windowName3);
}

struct TrackbarUserData
{
	TrackbarUserData(const cv::Mat &_img1, const cv::Mat &_img2, const std::string &_windowName)
	: img1(_img1), img2(_img2), windowName(_windowName)
	{}

	const cv::Mat &img1;
	const cv::Mat &img2;
	const std::string &windowName;
};

void on_trackbar(int pos, void *userData)
{
	if (!userData) return;

	const TrackbarUserData *data = reinterpret_cast<TrackbarUserData *>(userData);

	cv::Mat diff_img;
	cv::absdiff(data->img1, data->img2, diff_img);
	diff_img = diff_img > pos;

	cv::Mat filtered_img;
	data->img2.copyTo(filtered_img, diff_img);

	//
	cv::imshow(data->windowName, filtered_img);
}

void image_subtraction_with_trackbar()
{
	const std::string img1_name("machine_vision_data\\opencv\\table_only.jpg");
	const std::string img2_name("machine_vision_data\\opencv\\table_hand_01.jpg");
	//const std::string img2_name("machine_vision_data\\opencv\\table_hand_02.jpg");
	//const std::string img2_name("machine_vision_data\\opencv\\table_hand_03.jpg");
	//const std::string img2_name("machine_vision_data\\opencv\\table_hand_04.jpg");
	//const std::string img2_name("machine_vision_data\\opencv\\table_hand_05.jpg");

	const cv::Mat &img1 = cv::imread(img1_name, CV_LOAD_IMAGE_GRAYSCALE);
	const cv::Mat &img2 = cv::imread(img2_name, CV_LOAD_IMAGE_GRAYSCALE);
	if (img1.empty() || img2.empty())
	{
		std::cout << "fail to load image files" << std::endl;
		return;
	}

	const std::string windowName1("image subtraction - bg only");
	const std::string windowName2("image subtraction - fg + bg");
	const std::string windowName3("image subtraction - result");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName3, cv::WINDOW_AUTOSIZE);

	cv::Mat processed_img1, processed_img2;
#if 0
	canny(img1, processed_img1);
	canny(img2, processed_img2);
#elif 0
	sobel(img1, processed_img1);
	sobel(img2, processed_img2);
#else
	processed_img1 = img1;
	processed_img2 = img2;
#endif

	//processed_img1 = processed_img1 > 0;
	//processed_img2 = processed_img2 > 0;

	cv::imshow(windowName1, processed_img1);
	cv::imshow(windowName2, processed_img2);

	TrackbarUserData userData(processed_img1, processed_img2, windowName3);
	const int pos = 64;
	cv::createTrackbar("threshold", windowName3, (int *)&pos, 255, on_trackbar, (void *)&userData);
    on_trackbar(pos, (void *)&userData);

	const unsigned char key = cv::waitKey(0);
	if (27 == key)
		return;

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
	cv::destroyWindow(windowName3);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_operation()
{
	//local::image_handling();
	//local::show_image();
	//local::image_subtraction();
	local::image_subtraction_with_trackbar();
}

}  // namespace my_opencv
