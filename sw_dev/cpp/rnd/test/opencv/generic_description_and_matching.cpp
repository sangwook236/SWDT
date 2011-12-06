#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <list>


namespace {

void drawCorrespondences(const cv::Mat &img1, const std::vector<cv::KeyPoint> &features1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &features2, const std::vector<cv::DMatch> &desc_idx, cv::Mat &img_corr)
{
	img_corr.create(img1.rows + img2.rows, std::max(img1.cols, img2.cols), CV_8UC1);
	cv::cvtColor(img1, img_corr(cv::Rect(0, 0, img1.rows, img1.cols)), CV_GRAY2RGB);
	cv::cvtColor(img2, img_corr(cv::Rect(img1.rows, 0, img2.rows, img2.cols)), CV_GRAY2RGB);

	for (size_t i = 0; i < features1.size(); ++i)
	{
		cv::circle(img_corr, features1[i].pt, 3, CV_RGB(255, 0, 0));
	}

	for (size_t i = 0; i < features2.size(); ++i)
	{
		const cv::Point pt(cvRound(features2[i].pt.x + img1.rows), cvRound(features2[i].pt.y));
		cv::circle(img_corr, pt, 3, CV_RGB(255, 0, 0));
		cv::line(img_corr, features1[desc_idx[i].trainIdx].pt, pt, CV_RGB(0, 255, 0));
	}
}

}  // unnamed namespace

void generic_description_and_matching()
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

	// "Fern", "OneWay", "Vector"
	const std::string algorithm_name("FERN");
	const std::string algorithm_params_filename("opencv_data\\fern_params.xml");

	cv::Ptr<cv::GenericDescriptorMatcher> descriptorMatcher = cv::GenericDescriptorMatcher::create(algorithm_name, algorithm_params_filename);
	if (0 == descriptorMatcher)
	{
		std::cout << "cannot create descriptor" << std::endl;
		return;
	}

	//std::cout << "reading the images..." << std::endl;
	const cv::Mat &img1 = cv::imread(img1_name, CV_LOAD_IMAGE_GRAYSCALE);
	const cv::Mat &img2 = cv::imread(img2_name, CV_LOAD_IMAGE_GRAYSCALE);
	if (img1.empty() || img2.empty())
	{
		std::cout << "fail to load image files" << std::endl;
		return;
	}

	// extract keypoints
	cv::SURF keyPointExtractor(5.0e3);
	std::vector<cv::KeyPoint> keypoints1;

	std::cout << "extracting keypoints" << std::endl;
	keyPointExtractor(img1, cv::Mat(), keypoints1);
	std::cout << "\textracted " << keypoints1.size() << " keypoints from the first image" << std::endl;

	std::vector<cv::KeyPoint> keypoints2;
	keyPointExtractor(img2, cv::Mat(), keypoints2);
	std::cout << "\textracted " << keypoints2.size() << " keypoints from the second image" << std::endl;

	// find NN for each of keypoints2 in keypoints1
	std::cout << "finding nearest neighbors..." << std::endl;
	std::vector<cv::DMatch> matches2to1;
	descriptorMatcher->match(img2, keypoints2, img1, keypoints1, matches2to1);
	std::cout << "done" << std::endl;

	cv::Mat img_corr;
	drawCorrespondences(img1, keypoints1, img2, keypoints2, matches2to1, img_corr);

	const std::string windowName("generic description & matching");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowName, img_corr);
	cv::waitKey(0);
	cv::destroyWindow(windowName);
}