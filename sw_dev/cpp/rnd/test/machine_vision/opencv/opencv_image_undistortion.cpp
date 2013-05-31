//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <deque>
#include <string>
#include <iostream>


#define __USE_REMAP 1

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_undistortion()
{
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

	// [ref]
	//	Camera Calibration Toolbox for Matlab: http://www.vision.caltech.edu/bouguetj/calib_doc/
	//	http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

	//	In order to use the calibration results from Camera Calibration Toolbox for Matlab,
	//	a parameter for radial distrtortion, kc(5) has to be active, est_dist(5) = 1.

	// IR (left) to RGB (right)
#if 1
	const double fc_left[] = { 5.865259629841016e+02, 5.869119946890888e+02 };  // [pixel]
	const double cc_left[] = { 3.374030882445484e+02, 2.486851671279394e+02 };  // [pixel]
	const double alpha_c_left = 0.0;
	const double kc_left[] = { -1.191900881116634e-01, 4.770987342499287e-01, -2.359540729860922e-03, 6.980497982277759e-03, -4.910177215685992e-01 };

	const double fc_right[] = { 5.246059127053605e+02, 5.268025156310126e+02 };  // [pixel]
	const double cc_right[] = { 3.278147550011708e+02, 2.616335603485837e+02 };  // [pixel]
	const double alpha_c_right = 0.0;
	const double kc_right[] = { 2.877942533299389e-01, -1.183365423881459e+00, 8.819616313287803e-04, 3.334954093546241e-03, 1.924070186169354e+00 };

	const double rotVec[] = { -2.563731899937744e-03, 1.170121176120723e-02, 3.398860594633926e-03 };
	const double transVec[] = { 2.509097936007354e+01, 4.114956573893061e+00, -6.430074873604823e+00 };  // [mm]

	//
	cv::Mat K_left(3, 3, CV_64FC1, cv::Scalar::all(0)), K_right(3, 3, CV_64FC1, cv::Scalar::all(0));
	K_left.at<double>(0, 0) = fc_left[0];
	K_left.at<double>(0, 1) = alpha_c_left * fc_left[0];
	K_left.at<double>(0, 2) = cc_left[0];
	K_left.at<double>(1, 1) = fc_left[1];
	K_left.at<double>(1, 2) = cc_left[1];
	K_left.at<double>(2, 2) = 1.0;
	K_right.at<double>(0, 0) = fc_right[0];
	K_right.at<double>(0, 1) = alpha_c_right * fc_right[0];
	K_right.at<double>(0, 2) = cc_right[0];
	K_right.at<double>(1, 1) = fc_right[1];
	K_right.at<double>(1, 2) = cc_right[1];
	K_right.at<double>(2, 2) = 1.0;

	cv::Mat distCoeffs_left(1, 5, CV_64FC1, (void *)kc_left), distCoeffs_right(1, 5, CV_64FC1, (void *)kc_right);

    //const cv::Mat rvec = (cv::Mat_<double>(3,1) << rotVec[0], rotVec[1], rotVec[2]);
	const cv::Mat rvec(3, 1, CV_64FC1, (void *)rotVec);
	cv::Mat R;
    cv::Rodrigues(rvec, R);
    //const cv::Mat T = (cv::Mat_<double>(3,1) << transVec[0], transVec[1], transVec[2]);
    const cv::Mat T(3, 1, CV_64FC1, (void *)transVec);
#else
	const float fc_left[] = { 5.865259629841016e+02f, 5.869119946890888e+02f };  // [pixel]
	const float cc_left[] = { 3.374030882445484e+02f, 2.486851671279394e+02f };  // [pixel]
	const float alpha_c_left = 0.0f;
	const float kc_left[] = { -1.191900881116634e-01f, 4.770987342499287e-01f, -2.359540729860922e-03f, 6.980497982277759e-03f, -4.910177215685992e-01f };

	const float fc_right[] = { 5.246059127053605e+02f, 5.268025156310126e+02f };  // [pixel]
	const float cc_right[] = { 3.278147550011708e+02f, 2.616335603485837e+02f };  // [pixel]
	const float alpha_c_right = 0.0f;
	const float kc_right[] = { 2.877942533299389e-01f, -1.183365423881459e+00f, 8.819616313287803e-04f, 3.334954093546241e-03f, 1.924070186169354e+00f };
	//const float fc_right[] = { 5.2635228817969698e+002f, 5.2765917576898983e+002f };  // [pixel]
	//const float cc_right[] = { 3.2721575024118914e+002f, 2.6550336208783216e+002f };  // [pixel]
	//const float alpha_c_right = 0.0f;
	//const float kc_right[] = { 2.5703815534648017e-001f, -8.6596989999336349e-001f, 2.2803193915667845e-003f, 3.3064737839550973e-003f, 9.9706986903207828e-001f };

	const float rotVec[] = { -2.563731899937744e-03f, 1.170121176120723e-02f, 3.398860594633926e-03f };
	const float transVec[] = { 2.509097936007354e+01f, 4.114956573893061e+00f, -6.430074873604823e+00f };  // [mm]

	//
	cv::Mat K_left(3, 3, CV_32FC1, cv::Scalar::all(0)), K_right(3, 3, CV_32FC1, cv::Scalar::all(0));
	K_left.at<float>(0, 0) = fc_left[0];
	K_left.at<float>(0, 1) = alpha_c_left * fc_left[0];
	K_left.at<float>(0, 2) = cc_left[0];
	K_left.at<float>(1, 1) = fc_left[1];
	K_left.at<float>(1, 2) = cc_left[1];
	K_left.at<float>(2, 2) = 1.0f;
	K_right.at<float>(0, 0) = fc_right[0];
	K_right.at<float>(0, 1) = alpha_c_right * fc_right[0];
	K_right.at<float>(0, 2) = cc_right[0];
	K_right.at<float>(1, 1) = fc_right[1];
	K_right.at<float>(1, 2) = cc_right[1];
	K_right.at<float>(2, 2) = 1.0f;

	cv::Mat distCoeffs_left(1, 5, CV_32FC1, (void *)kc_left), distCoeffs_right(1, 5, CV_32FC1, (void *)kc_right);

    //const cv::Mat rvec = (cv::Mat_<float>(3,1) << rotVec[0], rotVec[1], rotVec[2]);
	const cv::Mat rvec(3, 1, CV_32FC1, (void *)rotVec);
	cv::Mat R;
    cv::Rodrigues(rvec, R);
    //const cv::Mat T = (cv::Mat_<float>(3,1) << transVec[0], transVec[1], transVec[2]);
    const cv::Mat T(3, 1, CV_32FC1, (void *)transVec);
#endif

	// undistort images
	if (false)
	{
#if __USE_REMAP
		cv::Mat rmap_left[2], rmap_right[2];
		cv::initUndistortRectifyMap(
			K_left, distCoeffs_left, cv::Mat(),
			cv::getOptimalNewCameraMatrix(K_left, distCoeffs_left, imageSize_left, 1, imageSize_left, 0),
			imageSize_left, CV_16SC2, rmap_left[0], rmap_left[1]
		);
		cv::initUndistortRectifyMap(
			K_right, distCoeffs_right, cv::Mat(),
			cv::getOptimalNewCameraMatrix(K_right, distCoeffs_right, imageSize_right, 1, imageSize_right, 0),
			imageSize_right, CV_16SC2, rmap_right[0], rmap_right[1]
		);
#endif

		cv::Mat img_left_after, img_right_after;
		cv::Mat img_left_after2;
		double minVal = 0.0, maxVal = 0.0;
		for (std::size_t k = 0; k < num_images; ++k)
		{
			const cv::Mat img_left_before(cv::imread(left_image_filenames[k], CV_LOAD_IMAGE_UNCHANGED));
			const cv::Mat img_right_before(cv::imread(right_image_filenames[k], CV_LOAD_IMAGE_COLOR));

#if __USE_REMAP
			cv::remap(img_left_before, img_left_after, rmap_left[0], rmap_left[1], cv::INTER_LINEAR);
			cv::remap(img_right_before, img_right_after, rmap_right[0], rmap_right[1], cv::INTER_LINEAR);
#else
			cv::undistort(img_left_before, img_left_after, K_left, distCoeffs_left, K_left);
			cv::undistort(img_right_before, img_right_after, K_right, distCoeffs_right, K_right);
#endif

			//
			cv::minMaxLoc(img_left_after, &minVal, &maxVal);

			img_left_after.convertTo(img_left_after2, CV_32FC1, 1.0f / (float)maxVal, 0.0f);

			cv::imshow("undistorted left image", img_left_after2);
			cv::imshow("undistorted right image", img_right_after);

			cv::waitKey(0);
		}
	}

	// rectify images
	if (true)
	{
		cv::Mat R_left, R_right, P_left, P_right, Q;
		cv::Rect validRoi_left, validRoi_right;

		cv::stereoRectify(
			K_left, distCoeffs_left,
			K_right, distCoeffs_right,
			imageSize_left, R, T, R_left, R_right, P_left, P_right, Q,
			cv::CALIB_ZERO_DISPARITY, 1, imageSize_left, &validRoi_left, &validRoi_right
		);

		// OpenCV can handle left-right or up-down camera arrangements
		const bool isVerticalStereo = std::fabs(P_right.at<double>(1, 3)) > std::fabs(P_right.at<double>(0, 3));

		cv::Mat rmap_left[2], rmap_right[2];
		cv::initUndistortRectifyMap(K_left, distCoeffs_left, R_left, P_left, imageSize_left, CV_16SC2, rmap_left[0], rmap_left[1]);
		cv::initUndistortRectifyMap(K_right, distCoeffs_right, R_right, P_right, imageSize_right, CV_16SC2, rmap_right[0], rmap_right[1]);

		cv::Mat img_left_after, img_right_after;
		cv::Mat cimg_left_after, cimg_right_after, img_left_after2;
		double minVal = 0.0, maxVal = 0.0;
		for (std::size_t k = 0; k < num_images; ++k)
		{
			const cv::Mat img_left_before(cv::imread(left_image_filenames[k], CV_LOAD_IMAGE_UNCHANGED));
			const cv::Mat img_right_before(cv::imread(right_image_filenames[k], CV_LOAD_IMAGE_COLOR));

			cv::remap(img_left_before, img_left_after, rmap_left[0], rmap_left[1], CV_INTER_LINEAR);
			cv::remap(img_right_before, img_right_after, rmap_right[0], rmap_right[1], CV_INTER_LINEAR);

#if 0
			std::ostringstream strm1, strm2;
			strm1 << "./machine_vision_data/opencv/image_undistortion/rectified_image_left_" << k << ".png";
			cv::imwrite(strm1.str(), img_left_after);
			strm2 << "./machine_vision_data/opencv/image_undistortion/rectified_image_right_" << k << ".png";
			cv::imwrite(strm2.str(), img_right_after);
#endif

			//
			cv::minMaxLoc(img_left_after, &minVal, &maxVal);

			img_left_after.convertTo(img_left_after2, CV_32FC1, 1.0f / (float)maxVal, 0.0f);

			cv::imshow("rectified left image", img_left_after2);
			cv::imshow("rectified right image", img_right_after);

			cv::waitKey(0);
		}
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
