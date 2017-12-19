//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>


namespace {
namespace local {

void draw_warped_roi(cv::Mat &image, const int width, const int height, cv::Mat &W)
{
#define HOMO_VECTOR(H, x, y)\
    H.at<float>(0,0) = (float)(x);\
    H.at<float>(1,0) = (float)(y);\
    H.at<float>(2,0) = 1.;

#define GET_HOMO_VALUES(X, x, y)\
    (x) = static_cast<float> (X.at<float>(0,0)/X.at<float>(2,0));\
    (y) = static_cast<float> (X.at<float>(1,0)/X.at<float>(2,0));

	cv::Point2f top_left, top_right, bottom_left, bottom_right;

	cv::Mat H(3, 1, CV_32F);
	cv::Mat U(3, 1, CV_32F);

	cv::Mat warp_mat(cv::Mat::eye(3, 3, CV_32F));

	for (int y = 0; y < W.rows; ++y)
		for (int x = 0; x < W.cols; ++x)
			warp_mat.at<float>(y, x) = W.at<float>(y, x);

	// Warp the corners of rectangle.
	// Top-left.
	HOMO_VECTOR(H, 1, 1);
	cv::gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, top_left.x, top_left.y);

	// Top-right.
	HOMO_VECTOR(H, width, 1);
	cv::gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, top_right.x, top_right.y);

	// Bottom-left.
	HOMO_VECTOR(H, 1, height);
	cv::gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, bottom_left.x, bottom_left.y);

	// Bottom-right.
	HOMO_VECTOR(H, width, height);
	cv::gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, bottom_right.x, bottom_right.y);

	// Draw the warped perimeter
	cv::line(image, top_left, top_right, cv::Scalar(255));
	cv::line(image, top_right, bottom_right, cv::Scalar(255));
	cv::line(image, bottom_right, bottom_left, cv::Scalar(255));
	cv::line(image, bottom_left, top_left, cv::Scalar(255));
}

// REF [file] >> ${OPENCV_HOME}/samples/cpp/image_alignment.cpp
void image_alignment_sample()
{
#if 0
	const std::string template_filename;
	const std::string input_filename("./data/machine_vision/opencv/fruits.jpg");
#elif 1
	//const std::string template_filename("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg");
	//const std::string input_filename("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_2.jpg");

	const std::string template_filename("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_1.jpg");
	const std::string input_filename("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_2.jpg");
#endif

	const size_t num_iterations = 100;
	const double epsilon = 0.0001;
	const int warp_type = cv::MOTION_HOMOGRAPHY;  // { cv::MOTION_TRANSLATION, cv::MOTION_EUCLIDEAN, cv::MOTION_AFFINE, cv::MOTION_HOMOGRAPHY }.

	const cv::Mat gray(cv::imread(input_filename, cv::IMREAD_GRAYSCALE));
	if (gray.empty())
	{
		std::cout << "Failed to load an image file: " << input_filename << std::endl;
		return;
	}

#if 0
	cv::Mat img;
	cv::resize(gray, img, cv::Size(216, 216));
#else
	cv::Mat img = gray.clone();
#endif

	cv::Mat template_img;
	if (!template_filename.empty())
	{
		template_img = cv::imread(template_filename, cv::IMREAD_GRAYSCALE);
		if (template_img.empty())
		{
			std::cout << "Failed to load an image file: " << template_filename << std::endl;
			return;
		}
	}
	else
	{
		// Apply random warp to input image.
		cv::Mat warp_ground;
		cv::RNG rng(cv::getTickCount());
		double angle;
		switch (warp_type)
		{
		case cv::MOTION_TRANSLATION:
			warp_ground = (cv::Mat_<float>(2, 3) << 1.0f, 0.0f, (rng.uniform(10.0f, 20.0f)),
				0.0f, 1.0f, (rng.uniform(10.0f, 20.0f)));
			cv::warpAffine(img, template_img, warp_ground, cv::Size(200, 200), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
			break;
		case cv::MOTION_EUCLIDEAN:
			angle = CV_PI / 30.0 + CV_PI * rng.uniform(-2.0, 2.0) / 180.0;
			warp_ground = (cv::Mat_<float>(2, 3) << (float)std::cos(angle), (float)-std::sin(angle), (rng.uniform(10.0f, 20.0f)),
				(float)std::sin(angle), (float)std::cos(angle), (rng.uniform(10.0f, 20.0f)));
			cv::warpAffine(img, template_img, warp_ground, cv::Size(200, 200), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
			break;
		case cv::MOTION_AFFINE:
			warp_ground = (cv::Mat_<float>(2, 3) << (1 - rng.uniform(-0.05f, 0.05f)),
				(rng.uniform(-0.03f, 0.03f)), (rng.uniform(10.0f, 20.0f)),
				(rng.uniform(-0.03f, 0.03f)), (1 - rng.uniform(-0.05f, 0.05f)),
				(rng.uniform(10.0f, 20.0f)));
			cv::warpAffine(img, template_img, warp_ground, cv::Size(200, 200), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
			break;
		case cv::MOTION_HOMOGRAPHY:
			warp_ground = (cv::Mat_<float>(3, 3) << (1 - rng.uniform(-0.05f, 0.05f)),
				(rng.uniform(-0.03f, 0.03f)), (rng.uniform(10.0f, 20.0f)),
				(rng.uniform(-0.03f, 0.03f)), (1 - rng.uniform(-0.05f, 0.05f)), (rng.uniform(10.0f, 20.0f)),
				(rng.uniform(0.0001f, 0.0003f)), (rng.uniform(0.0001f, 0.0003f)), 1.0f);
			cv::warpPerspective(img, template_img, warp_ground, cv::Size(200, 200), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
			break;
		}
	}

	// Initialize a warp matrix.
	cv::Mat warp_matrix;
	if (cv::MOTION_HOMOGRAPHY == warp_type)
		warp_matrix = cv::Mat::eye(3, 3, CV_32F);
	else
		warp_matrix = cv::Mat::eye(2, 3, CV_32F);

	//
	const double tic_init = (double)cv::getTickCount();
	const double cc = cv::findTransformECC(template_img, img, warp_matrix, warp_type,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, num_iterations, epsilon));

	if (-1 == cc)
	{
		std::cerr << "The execution was interrupted. The correlation value is going to be minimized." << std::endl;
		std::cerr << "Check the warp initialization and/or the size of images." << std::endl;
	}

	const double toc_final = (double)cv::getTickCount();
	const double total_time = (toc_final - tic_init) / (cv::getTickFrequency());
	std::cout << "Alignment time (" << warp_type << " transformation): " << total_time << " sec." << std::endl;
	std::cout << "Final correlation: " << cc << std::endl;

	// Warp image.
	cv::Mat warped_img(template_img.rows, template_img.cols, CV_32FC1);
	if (cv::MOTION_HOMOGRAPHY == warp_type)
		cv::warpPerspective(img, warped_img, warp_matrix, warped_img.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
	else
		cv::warpAffine(img, warped_img, warp_matrix, warped_img.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);

	// Show result.
	// Draw boundaries of corresponding regions;
	cv::Mat identity_matrix(cv::Mat::eye(3, 3, CV_32F));
	draw_warped_roi(img, template_img.cols - 2, template_img.rows - 2, warp_matrix);
	draw_warped_roi(template_img, template_img.cols - 2, template_img.rows - 2, identity_matrix);

	cv::Mat error_img;
	cv::subtract(template_img, warped_img, error_img);
	double max_of_error;
	cv::minMaxLoc(error_img, NULL, &max_of_error);

	cv::imshow("Image warping - Original", img);
	cv::imshow("Image warping - Template", template_img);
	cv::imshow("Image warping - Warped", warped_img);
	cv::imshow("Image warping - Error (black: no error)", cv::abs(error_img) * 255 / max_of_error);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_alignment()
{
	local::image_alignment_sample();
}

}  // namespace my_opencv
