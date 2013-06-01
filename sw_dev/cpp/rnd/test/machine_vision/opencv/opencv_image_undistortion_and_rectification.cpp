//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <deque>
#include <string>
#include <iostream>


#define __USE_REMAP 1

namespace {
namespace local {

// undistort images
void undistort_images(const std::vector<cv::Mat> &input_images, std::vector<cv::Mat> &output_images,
	const cv::Size &imageSize, const cv::Mat &K, const cv::Mat &distCoeffs)
{
#if __USE_REMAP
	cv::Mat rmap[2];
	cv::initUndistortRectifyMap(
		K, distCoeffs, cv::Mat(),
		cv::getOptimalNewCameraMatrix(K, distCoeffs, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, rmap[0], rmap[1]
	);
#endif

	cv::Mat img_after;
	for (std::vector<cv::Mat>::const_iterator cit = input_images.begin(); cit != input_images.end(); ++cit)
	{
		const cv::Mat &img_before = *cit;

#if __USE_REMAP
		cv::remap(img_before, img_after, rmap[0], rmap[1], cv::INTER_LINEAR);
#else
		cv::undistort(img_before, img_after, K, distCoeffs, K);
#endif

		output_images.push_back(img_after);
	}
}

void rectify_images(const std::vector<cv::Mat> &input_images_left, const std::vector<cv::Mat> &input_images_right, std::vector<cv::Mat> &output_images_left, std::vector<cv::Mat> &output_images_right,
	const cv::Size &imageSize_left, const cv::Size &imageSize_right,
	const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &distCoeffs_left, const cv::Mat &distCoeffs_right,
	const cv::Mat &R, const cv::Mat &T)
{
	const std::size_t num_images = input_images_left.size();
	if (input_images_right.size() != num_images)
	{
		std::cerr << "the numbers of left & right input images are different" << std::endl;
		return;
	}

	cv::Mat R_left, R_right, P_left, P_right, Q;
	cv::Rect validRoi_left, validRoi_right;

	cv::stereoRectify(
		K_left, distCoeffs_left,
		K_right, distCoeffs_right,
		imageSize_left, R, T, R_left, R_right, P_left, P_right, Q,
		cv::CALIB_ZERO_DISPARITY, 1, imageSize_left, &validRoi_left, &validRoi_right
	);

	// OpenCV can handle left-right or up-down camera arrangements
	//const bool isVerticalStereo = std::fabs(P_right.at<double>(1, 3)) > std::fabs(P_right.at<double>(0, 3));

	cv::Mat rmap_left[2], rmap_right[2];
	cv::initUndistortRectifyMap(K_left, distCoeffs_left, R_left, P_left, imageSize_left, CV_16SC2, rmap_left[0], rmap_left[1]);
	cv::initUndistortRectifyMap(K_right, distCoeffs_right, R_right, P_right, imageSize_right, CV_16SC2, rmap_right[0], rmap_right[1]);

	cv::Mat img_left_after, img_right_after;
	for (std::size_t k = 0; k < num_images; ++k)
	{
		const cv::Mat &img_left_before = input_images_left[k];
		const cv::Mat &img_right_before = input_images_right[k];

		cv::remap(img_left_before, img_left_after, rmap_left[0], rmap_left[1], CV_INTER_LINEAR);
		cv::remap(img_right_before, img_right_after, rmap_right[0], rmap_right[1], CV_INTER_LINEAR);

#if 0
		std::ostringstream strm1, strm2;
		strm1 << "./machine_vision_data/opencv/image_undistortion/rectified_image_left_" << k << ".png";
		cv::imwrite(strm1.str(), img_left_after);
		strm2 << "./machine_vision_data/opencv/image_undistortion/rectified_image_right_" << k << ".png";
		cv::imwrite(strm2.str(), img_right_after);
#endif

		output_images_left.push_back(img_left_after);
		output_images_right.push_back(img_right_after);
	}
}

void load_kinect_sensor_parameters(
	cv::Mat &K_ir, cv::Mat &distCoeffs_ir, cv::Mat &K_rgb, cv::Mat &distCoeffs_rgb,
	cv::Mat &R_ir_to_rgb, cv::Mat &T_ir_to_rgb)
{
	// [ref]
	//	Camera Calibration Toolbox for Matlab: http://www.vision.caltech.edu/bouguetj/calib_doc/
	//	http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

	//	In order to use the calibration results from Camera Calibration Toolbox for Matlab,
	//	a parameter for radial distrtortion, kc(5) has to be active, est_dist(5) = 1.

	// IR (left) to RGB (right)
#if 1
	const double fc_ir[] = { 5.865259629841016e+02, 5.869119946890888e+02 };  // [pixel]
	const double cc_ir[] = { 3.374030882445484e+02, 2.486851671279394e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	const double kc_ir[] = { -1.191900881116634e-01, 4.770987342499287e-01, -2.359540729860922e-03, 6.980497982277759e-03, -4.910177215685992e-01 };

	const double fc_rgb[] = { 5.246059127053605e+02, 5.268025156310126e+02 };  // [pixel]
	const double cc_rgb[] = { 3.278147550011708e+02, 2.616335603485837e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	const double kc_rgb[] = { 2.877942533299389e-01, -1.183365423881459e+00, 8.819616313287803e-04, 3.334954093546241e-03, 1.924070186169354e+00 };

	const double rotVec[] = { -2.563731899937744e-03, 1.170121176120723e-02, 3.398860594633926e-03 };
	const double transVec[] = { 2.509097936007354e+01, 4.114956573893061e+00, -6.430074873604823e+00 };  // [mm]

	//
	K_ir = cv::Mat::zeros(3, 3, CV_64FC1);
	K_ir.at<double>(0, 0) = fc_ir[0];
	K_ir.at<double>(0, 1) = alpha_c_ir * fc_ir[0];
	K_ir.at<double>(0, 2) = cc_ir[0];
	K_ir.at<double>(1, 1) = fc_ir[1];
	K_ir.at<double>(1, 2) = cc_ir[1];
	K_ir.at<double>(2, 2) = 1.0;
	K_rgb = cv::Mat::zeros(3, 3, CV_64FC1);
	K_rgb.at<double>(0, 0) = fc_rgb[0];
	K_rgb.at<double>(0, 1) = alpha_c_rgb * fc_rgb[0];
	K_rgb.at<double>(0, 2) = cc_rgb[0];
	K_rgb.at<double>(1, 1) = fc_rgb[1];
	K_rgb.at<double>(1, 2) = cc_rgb[1];
	K_rgb.at<double>(2, 2) = 1.0;

	distCoeffs_ir = cv::Mat(1, 5, CV_64FC1, (void *)kc_ir);
	distCoeffs_rgb = cv::Mat(1, 5, CV_64FC1, (void *)kc_rgb);

    cv::Rodrigues(cv::Mat(3, 1, CV_64FC1, (void *)rotVec), R_ir_to_rgb);
    T_ir_to_rgb = cv::Mat(3, 1, CV_64FC1, (void *)transVec);
#else
	const float fc_ir[] = { 5.865259629841016e+02f, 5.869119946890888e+02f };  // [pixel]
	const float cc_ir[] = { 3.374030882445484e+02f, 2.486851671279394e+02f };  // [pixel]
	const float alpha_c_ir = 0.0f;
	const float kc_ir[] = { -1.191900881116634e-01f, 4.770987342499287e-01f, -2.359540729860922e-03f, 6.980497982277759e-03f, -4.910177215685992e-01f };

	const float fc_rgb[] = { 5.246059127053605e+02f, 5.268025156310126e+02f };  // [pixel]
	const float cc_rgb[] = { 3.278147550011708e+02f, 2.616335603485837e+02f };  // [pixel]
	const float alpha_c_rgb = 0.0f;
	const float kc_rgb[] = { 2.877942533299389e-01f, -1.183365423881459e+00f, 8.819616313287803e-04f, 3.334954093546241e-03f, 1.924070186169354e+00f };
	//const float fc_rgb[] = { 5.2635228817969698e+002f, 5.2765917576898983e+002f };  // [pixel]
	//const float cc_rgb[] = { 3.2721575024118914e+002f, 2.6550336208783216e+002f };  // [pixel]
	//const float alpha_c_rgb = 0.0f;
	//const float kc_rgb[] = { 2.5703815534648017e-001f, -8.6596989999336349e-001f, 2.2803193915667845e-003f, 3.3064737839550973e-003f, 9.9706986903207828e-001f };

	const float rotVec[] = { -2.563731899937744e-03f, 1.170121176120723e-02f, 3.398860594633926e-03f };
	const float transVec[] = { 2.509097936007354e+01f, 4.114956573893061e+00f, -6.430074873604823e+00f };  // [mm]

	//
	K_ir = cv::Mat::zeros(3, 3, CV_32FC1);
	K_ir.at<float>(0, 0) = fc_ir[0];
	K_ir.at<float>(0, 1) = alpha_c_ir * fc_ir[0];
	K_ir.at<float>(0, 2) = cc_ir[0];
	K_ir.at<float>(1, 1) = fc_ir[1];
	K_ir.at<float>(1, 2) = cc_ir[1];
	K_ir.at<float>(2, 2) = 1.0f;
	K_rgb = cv::Mat::zeros(3, 3, CV_32FC1);
	K_rgb.at<float>(0, 0) = fc_rgb[0];
	K_rgb.at<float>(0, 1) = alpha_c_rgb * fc_rgb[0];
	K_rgb.at<float>(0, 2) = cc_rgb[0];
	K_rgb.at<float>(1, 1) = fc_rgb[1];
	K_rgb.at<float>(1, 2) = cc_rgb[1];
	K_rgb.at<float>(2, 2) = 1.0f;

	distCoeffs_ir = cv::Mat(1, 5, CV_32FC1, (void *)kc_ir);
	distCoeffs_rgb = cv::Mat(1, 5, CV_32FC1, (void *)kc_rgb);

    cv::Rodrigues(cv::Mat(3, 1, CV_32FC1, (void *)rotVec), R_ir_to_rgb);
    T_ir_to_rgb = cv::Mat(3, 1, CV_32FC1, (void *)transVec);
#endif
}

void kinect_image_undistortion()
{
	// prepare input images
	std::deque<std::string> left_image_filenames;
	left_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130530T103805.png");
	left_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130531T023152.png");
	left_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130531T023346.png");
	left_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130531T023359.png");

	std::deque<std::string> right_image_filenames;
	right_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130530T103805.png");
	right_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023152.png");
	right_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023346.png");
	right_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023359.png");

	const std::size_t num_images = 4;
	const cv::Size imageSize_left(640, 480), imageSize_right(640, 480);

	std::vector<cv::Mat> input_images_left, input_images_right;
	for (std::size_t k = 0; k < num_images; ++k)
	{
		input_images_left.push_back(cv::imread(left_image_filenames[k], CV_LOAD_IMAGE_UNCHANGED));
		input_images_right.push_back(cv::imread(right_image_filenames[k], CV_LOAD_IMAGE_COLOR));
	}

	// get the camera parameters of a Kinect sensor.
	cv::Mat K_left, K_right;
	cv::Mat distCoeffs_left, distCoeffs_right;
	cv::Mat R, T;
	load_kinect_sensor_parameters(K_left, distCoeffs_left, K_right, distCoeffs_right, R, T);

	// undistort images (left)
	{
		std::vector<cv::Mat> output_images;
		undistort_images(input_images_left, output_images, imageSize_left, K_left, distCoeffs_left);

		// show results
		cv::Mat img_after;
		double minVal = 0.0, maxVal = 0.0;
		for (std::vector<cv::Mat>::const_iterator cit = output_images.begin(); cit != output_images.end(); ++cit)
		{
			cv::minMaxLoc(*cit, &minVal, &maxVal);
			cit->convertTo(img_after, CV_32FC1, 1.0f / (float)maxVal, 0.0f);

			cv::imshow("undistorted left image", img_after);

			cv::waitKey(0);
		}
	}

	// undistort images (right)
	{
		std::vector<cv::Mat> output_images;
		undistort_images(input_images_right, output_images, imageSize_right, K_right, distCoeffs_right);

		// show results
		for (std::vector<cv::Mat>::const_iterator cit = output_images.begin(); cit != output_images.end(); ++cit)
		{
			cv::imshow("undistorted right image", *cit);

			cv::waitKey(0);
		}
	}

	cv::destroyAllWindows();
}

void kinect_image_rectification()
{
	// prepare input images
	std::deque<std::string> left_image_filenames;
	left_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130530T103805.png");
	left_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130531T023152.png");
	left_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130531T023346.png");
	left_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130531T023359.png");

	std::deque<std::string> right_image_filenames;
	right_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130530T103805.png");
	right_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023152.png");
	right_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023346.png");
	right_image_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023359.png");

	const std::size_t num_images = 4;
	const cv::Size imageSize_left(640, 480), imageSize_right(640, 480);

	std::vector<cv::Mat> input_images_left, input_images_right;
	for (std::size_t k = 0; k < num_images; ++k)
	{
		input_images_left.push_back(cv::imread(left_image_filenames[k], CV_LOAD_IMAGE_UNCHANGED));
		input_images_right.push_back(cv::imread(right_image_filenames[k], CV_LOAD_IMAGE_COLOR));
	}

	// get the camera parameters of a Kinect sensor.
	cv::Mat K_left, K_right;
	cv::Mat distCoeffs_left, distCoeffs_right;
	cv::Mat R, T;
	load_kinect_sensor_parameters(K_left, distCoeffs_left, K_right, distCoeffs_right, R, T);

	// rectify images
	std::vector<cv::Mat> output_images_left, output_images_right;
	rectify_images(
		input_images_left, input_images_right, output_images_left, output_images_right,
		imageSize_left, imageSize_right,
		K_left, K_right, distCoeffs_left, distCoeffs_right, R, T
	);

	// show results
	cv::Mat img_left_after2;
	double minVal = 0.0, maxVal = 0.0;
	for (std::size_t k = 0; k < num_images; ++k)
	{
		const cv::Mat &img_left_after = output_images_left[k];
		const cv::Mat &img_right_after = output_images_right[k];

		//
		cv::minMaxLoc(img_left_after, &minVal, &maxVal);
		img_left_after.convertTo(img_left_after2, CV_32FC1, 1.0f / (float)maxVal, 0.0f);

		cv::imshow("rectified left image", img_left_after2);
		cv::imshow("rectified right image", img_right_after);

		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_undistortion()
{
	local::kinect_image_undistortion();
}

void image_rectification()
{
	local::kinect_image_rectification();
}

}  // namespace my_opencv
