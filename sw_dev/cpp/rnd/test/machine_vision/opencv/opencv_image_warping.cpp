//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${OPENCV_HOME}/samples/cpp/warpPerspective_demo.cpp
void warp_perspective_demo()
{
	const std::string filename("./data/machine_vision/right.jpg");

	const cv::Mat rgb(cv::imread(filename, cv::IMREAD_COLOR));
	if (rgb.empty())
	{
		std::cout << "Failed to load an image file: " << filename << std::endl;
		return;
	}

	std::vector<cv::Point2f> roi_corners;
	roi_corners.push_back(cv::Point2f((float)(rgb.cols / 1.70f), (float)(rgb.rows / 4.20f)));
	roi_corners.push_back(cv::Point2f((float)(rgb.cols / 1.15f), (float)(rgb.rows / 3.32f)));
	roi_corners.push_back(cv::Point2f((float)(rgb.cols / 1.33f), (float)(rgb.rows / 1.10f)));
	roi_corners.push_back(cv::Point2f((float)(rgb.cols / 1.93f), (float)(rgb.rows / 1.36f)));

	std::vector<cv::Point2f> dst_corners(4);
	dst_corners[0].x = 0.0f;
	dst_corners[0].y = 0.0f;
	dst_corners[1].x = (float)std::max(cv::norm(roi_corners[0] - roi_corners[1]), cv::norm(roi_corners[2] - roi_corners[3]));
	dst_corners[1].y = 0.0f;
	dst_corners[2].x = (float)std::max(cv::norm(roi_corners[0] - roi_corners[1]), cv::norm(roi_corners[2] - roi_corners[3]));
	dst_corners[2].y = (float)std::max(cv::norm(roi_corners[1] - roi_corners[2]), cv::norm(roi_corners[3] - roi_corners[0]));
	dst_corners[3].x = 0.0f;
	dst_corners[3].y = (float)std::max(cv::norm(roi_corners[1] - roi_corners[2]), cv::norm(roi_corners[3] - roi_corners[0]));

	const cv::String labels[4] = { "TL", "TR", "BR", "BL" };

	cv::Mat dst = rgb.clone();
	for (int i = 0; i < 4; ++i)
	{
		cv::line(dst, roi_corners[i], roi_corners[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
		cv::circle(dst, roi_corners[i], 5, cv::Scalar(0, 255, 0), 3);
		cv::putText(dst, labels[i].c_str(), roi_corners[i], cv::QT_FONT_NORMAL, 0.8, cv::Scalar(255, 0, 0), 2);
	}

	const cv::Size warped_image_size(cvRound(dst_corners[2].x), cvRound(dst_corners[2].y));

	// Compute homography.
	const cv::Mat H = cv::findHomography(roi_corners, dst_corners);

	// Warp image.
	cv::Mat warped_image;
	cv::warpPerspective(rgb, warped_image, H, warped_image_size);

	// Show result.
	cv::imshow("Image warping - Original", dst);
	cv::imshow("Image warping - Warped", warped_image);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

// REF [file] >> ${OPENCV_HOME}/samples/cpp/warpPerspective_demo.cpp
void auto_9_view()
{
#if 0
	const std::string filename("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_1.jpg");
	std::vector<cv::Point2f> roi_corners;
	roi_corners.push_back(cv::Point2f(82.0f, 102.0f));
	roi_corners.push_back(cv::Point2f(251.0f, 102.0f));
	roi_corners.push_back(cv::Point2f(251.0f, 151.0f));
	roi_corners.push_back(cv::Point2f(82.0f, 150.0f));
#elif 1
	const std::string filename("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_2.jpg");
	std::vector<cv::Point2f> roi_corners;
	roi_corners.push_back(cv::Point2f(127.0f, 73.0f));
	roi_corners.push_back(cv::Point2f(245.0f, 159.0f));
	roi_corners.push_back(cv::Point2f(199.0f, 194.0f));
	roi_corners.push_back(cv::Point2f(81.0f, 105.0f));
#endif

	const cv::Mat rgb(cv::imread(filename, cv::IMREAD_COLOR));
	if (rgb.empty())
	{
		std::cout << "Failed to load an image file: " << filename << std::endl;
		return;
	}

	std::vector<cv::Point2f> dst_corners;
	dst_corners.push_back(cv::Point2f(0.0f, 0.0f));
	dst_corners.push_back(cv::Point2f(176.0f, 0.0f));
	dst_corners.push_back(cv::Point2f(176.0f, 70.0f));
	dst_corners.push_back(cv::Point2f(0.0f, 70.0f));

	const cv::String labels[4] = { "TL", "TR", "BR", "BL" };

	cv::Mat dst = rgb.clone();
	for (int i = 0; i < 4; ++i)
	{
		cv::line(dst, roi_corners[i], roi_corners[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
		cv::circle(dst, roi_corners[i], 5, cv::Scalar(0, 255, 0), 3);
		cv::putText(dst, labels[i].c_str(), roi_corners[i], cv::QT_FONT_NORMAL, 0.8, cv::Scalar(255, 0, 0), 2);
	}

	const cv::Size warped_image_size(cvRound(dst_corners[2].x) * 2, cvRound(dst_corners[2].y) * 2);

	// Compute homography.
	const cv::Mat H = cv::findHomography(roi_corners, dst_corners);

	// Warp image.
	cv::Mat warped_image;
	cv::warpPerspective(rgb, warped_image, H, warped_image_size, cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar());

	// Show result.
	cv::imshow("Image warping - Original", dst);
	cv::imshow("Image warping - Warped", warped_image);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_warping()
{
	//local::warp_perspective_demo();

	// Application.
	local::auto_9_view();
}

}  // namespace my_opencv
