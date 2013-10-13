//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>


#define __USE_OPENCV_REMAP 1

namespace {
namespace local {

// [ref] readStringList() in opencv_camera_calibration.cpp
bool readStringList(const std::string &filename, std::vector<std::string> &l)
{
	l.resize(0);
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;

	cv::FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != cv::FileNode::SEQ)
		return false;

	cv::FileNodeIterator it = n.begin(), it_end = n.end();
	for ( ; it != it_end; ++it)
		l.push_back((std::string)*it);

	return true;
}

// [ref] undistort_images_using_opencv() in opencv_image_undistortion.cpp
void undistort_images_using_opencv(
	const std::vector<cv::Mat> &input_images, std::vector<cv::Mat> &output_images,
	const cv::Size &imageSize, const cv::Mat &K, const cv::Mat &distCoeffs
)
{
#if __USE_OPENCV_REMAP
	cv::Mat rmap[2];
	cv::initUndistortRectifyMap(
		K, distCoeffs, cv::Mat(),
		cv::getOptimalNewCameraMatrix(K, distCoeffs, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, rmap[0], rmap[1]
	);
#endif

	for (std::vector<cv::Mat>::const_iterator cit = input_images.begin(); cit != input_images.end(); ++cit)
	{
		const cv::Mat &img_before = *cit;

		cv::Mat img_after;
#if __USE_OPENCV_REMAP
		cv::remap(img_before, img_after, rmap[0], rmap[1], cv::INTER_LINEAR);
#else
		cv::undistort(img_before, img_after, K, distCoeffs, K);
#endif

		output_images.push_back(img_after);
	}
}

// [ref] undistort_images_using_formula() in opencv_image_undistortion.cpp
template <typename T>
void undistort_images_using_formula(
	const std::vector<cv::Mat> &input_images, std::vector<cv::Mat> &output_images,
	const cv::Size &imageSize, const cv::Mat &K, const cv::Mat &distCoeffs
)
{
	// [ref] "Joint Depth and Color Camera Calibration with Distortion Correction", D. Herrera C., J. Kannala, & J. Heikkila, TPAMI, 2012

	// homogeneous image coordinates: zero-based coordinates
	cv::Mat IC_homo(3, imageSize.height * imageSize.width, CV_64FC1, cv::Scalar::all(1));
	{
#if 0
		// 0 0 0 ...   0 1 1 1 ...   1 ... 639 639 639 ... 639
		// 0 1 2 ... 479 0 1 2 ... 479 ...   0   1   2 ... 479

		cv::Mat arr(1, imageSize.height, CV_64FC1);
		for (int i = 0; i < imageSize.height; ++i)
			arr.at<double>(0, i) = (double)i;

		for (int i = 0; i < imageSize.width; ++i)
		{
			IC_homo(cv::Range(0, 1), cv::Range(i * imageSize.height, (i + 1) * imageSize.height)).setTo(cv::Scalar::all(i));
			arr.copyTo(IC_homo(cv::Range(1, 2), cv::Range(i * imageSize.height, (i + 1) * imageSize.height)));
		}
#else
		// 0 1 2 ... 639 0 1 2 ... 639 ...   0   1   2 ... 639
		// 0 0 0 ...   0 1 1 1 ...   1 ... 479 479 479 ... 479

		cv::Mat arr(1, imageSize.width, CV_64FC1);
		for (int i = 0; i < imageSize.width; ++i)
			arr.at<double>(0, i) = (double)i;

		for (int i = 0; i < imageSize.height; ++i)
		{
			arr.copyTo(IC_homo(cv::Range(0, 1), cv::Range(i * imageSize.width, (i + 1) * imageSize.width)));
			IC_homo(cv::Range(1, 2), cv::Range(i * imageSize.width, (i + 1) * imageSize.width)).setTo(cv::Scalar::all(i));
		}
#endif
	}

	// homogeneous normalized camera coordinates
	const cv::Mat CC_norm(K.inv() * IC_homo);

	// apply distortion
	cv::Mat IC_homo_undist;
	{
		//const cv::Mat xn(CC_norm(cv::Range(0,1), cv::Range::all()));
		const cv::Mat xn(CC_norm(cv::Range(0,1), cv::Range::all()) / CC_norm(cv::Range(2,3), cv::Range::all()));
		//const cv::Mat yn(CC_norm(cv::Range(1,2), cv::Range::all()));
		const cv::Mat yn(CC_norm(cv::Range(1,2), cv::Range::all()) / CC_norm(cv::Range(2,3), cv::Range::all()));

		const cv::Mat xn2(xn.mul(xn));
		const cv::Mat yn2(yn.mul(yn));
		const cv::Mat xnyn(xn.mul(yn));
		const cv::Mat r2(xn2 + yn2);
		const cv::Mat r4(r2.mul(r2));
		const cv::Mat r6(r4.mul(r2));

		const double &k1 = distCoeffs.at<double>(0);
		const double &k2 = distCoeffs.at<double>(1);
		const double &k3 = distCoeffs.at<double>(2);
		const double &k4 = distCoeffs.at<double>(3);
		const double &k5 = distCoeffs.at<double>(4);

		const cv::Mat xg(2.0 * k3 * xnyn + k4 * (r2 + 2.0 * xn2));
		const cv::Mat yg(k3 * (r2 + 2.0 * yn2) + 2.0 * k4 * xnyn);

		const cv::Mat coeff(1.0 + k1 * r2 + k2 * r4 + k5 * r6);
		cv::Mat xk(3, imageSize.height * imageSize.width, CV_64FC1, cv::Scalar::all(1));
		cv::Mat(coeff.mul(xn) + xg).copyTo(xk(cv::Range(0,1), cv::Range::all()));
		cv::Mat(coeff.mul(yn) + yg).copyTo(xk(cv::Range(1,2), cv::Range::all()));

		IC_homo_undist = K * xk;
	}

	for (std::vector<cv::Mat>::const_iterator cit = input_images.begin(); cit != input_images.end(); ++cit)
	{
		const cv::Mat &img_before = *cit;

		cv::Mat img_after(img_before.size(), img_before.type(), cv::Scalar::all(0));
		for (int idx = 0; idx < imageSize.height*imageSize.width; ++idx)
		{
#if 0
			// don't apply interpolation.
			const int &cc_new = cvRound(IC_homo.at<double>(0,idx));
			const int &rr_new = cvRound(IC_homo.at<double>(1,idx));
			const int &cc = cvRound(IC_homo_undist.at<double>(0,idx));
			const int &rr = cvRound(IC_homo_undist.at<double>(1,idx));
			if (0 <= cc && cc < imageSize.width && 0 <= rr && rr < imageSize.height)
			{
				// TODO [check] >> why is the code below correctly working?
				//img_after.at<T>(rr, cc) = img_before.at<T>(rr_new, cc_new);
				img_after.at<T>(rr_new, cc_new) = img_before.at<T>(rr, cc);
			}
#else
			// apply interpolation.

			// TODO [enhance] >> speed up.

			const int &cc_new = cvRound(IC_homo.at<double>(0,idx));
			const int &rr_new = cvRound(IC_homo.at<double>(1,idx));

			const double &cc = IC_homo_undist.at<double>(0,idx);
			const double &rr = IC_homo_undist.at<double>(1,idx);
			const int cc_0 = cvFloor(cc), cc_1 = cc_0 + 1;
			const int rr_0 = cvFloor(rr), rr_1 = rr_0 + 1;
			const double alpha_cc = cc - cc_0, alpha_rr = rr - rr_0;
			if (0 <= cc_0 && cc_0 < imageSize.width - 1 && 0 <= rr_0 && rr_0 < imageSize.height - 1)
			{
				img_after.at<T>(rr_new, cc_new) =
					(T)((1.0 - alpha_rr) * (1.0 - alpha_cc) * img_before.at<T>(rr_0, cc_0) +
					(1.0 - alpha_rr) * alpha_cc * img_before.at<T>(rr_0, cc_1) +
					alpha_rr * (1.0 - alpha_cc) * img_before.at<T>(rr_1, cc_0) +
					alpha_rr * alpha_cc * img_before.at<T>(rr_1, cc_1));
			}
#endif
		}

		output_images.push_back(img_after);
	}
}

void rectify_images_using_opencv(
	const std::vector<cv::Mat> &input_images_left, const std::vector<cv::Mat> &input_images_right, std::vector<cv::Mat> &output_images_left, std::vector<cv::Mat> &output_images_right,
	const cv::Size &imageSize_left, const cv::Size &imageSize_right,
	const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &distCoeffs_left, const cv::Mat &distCoeffs_right,
	const cv::Mat &R, const cv::Mat &T
)
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

	for (std::size_t k = 0; k < num_images; ++k)
	{
		const cv::Mat &img_left_before = input_images_left[k];
		const cv::Mat &img_right_before = input_images_right[k];

		cv::Mat img_left_after, img_right_after;
		cv::remap(img_left_before, img_left_after, rmap_left[0], rmap_left[1], CV_INTER_LINEAR);
		cv::remap(img_right_before, img_right_after, rmap_right[0], rmap_right[1], CV_INTER_LINEAR);

		output_images_left.push_back(img_left_after);
		output_images_right.push_back(img_right_after);
	}
}

// IR (left) to RGB (right)
void rectify_kinect_images_from_IR_to_RGB_using_depth(
	const std::vector<cv::Mat> &input_images_left, const std::vector<cv::Mat> &input_images_right, std::vector<cv::Mat> &output_images_left, std::vector<cv::Mat> &output_images_right,
	const cv::Size &imageSize_left, const cv::Size &imageSize_right,
	const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &R, const cv::Mat &T
)
{
	const std::size_t num_images = input_images_left.size();
	if (input_images_right.size() != num_images)
	{
		std::cerr << "the numbers of left & right input images are different" << std::endl;
		return;
	}

	// homogeneous image coordinates (left): zero-based coordinates
	cv::Mat IC_homo_left(3, imageSize_left.height * imageSize_left.width, CV_64FC1, cv::Scalar::all(1));
	{
#if 0
		// 0 0 0 ...   0 1 1 1 ...   1 ... 639 639 639 ... 639
		// 0 1 2 ... 479 0 1 2 ... 479 ...   0   1   2 ... 479

		cv::Mat arr(1, imageSize_left.height, CV_64FC1);
		for (int i = 0; i < imageSize_left.height; ++i)
			arr.at<double>(0, i) = (double)i;

		for (int i = 0; i < imageSize_left.width; ++i)
		{
			IC_homo_left(cv::Range(0, 1), cv::Range(i * imageSize_left.height, (i + 1) * imageSize_left.height)).setTo(cv::Scalar::all(i));
			arr.copyTo(IC_homo_left(cv::Range(1, 2), cv::Range(i * imageSize_left.height, (i + 1) * imageSize_left.height)));
		}
#else
		// 0 1 2 ... 639 0 1 2 ... 639 ...   0   1   2 ... 639
		// 0 0 0 ...   0 1 1 1 ...   1 ... 479 479 479 ... 479

		cv::Mat arr(1, imageSize_left.width, CV_64FC1);
		for (int i = 0; i < imageSize_left.width; ++i)
			arr.at<double>(0, i) = (double)i;

		for (int i = 0; i < imageSize_left.height; ++i)
		{
			arr.copyTo(IC_homo_left(cv::Range(0, 1), cv::Range(i * imageSize_left.width, (i + 1) * imageSize_left.width)));
			IC_homo_left(cv::Range(1, 2), cv::Range(i * imageSize_left.width, (i + 1) * imageSize_left.width)).setTo(cv::Scalar::all(i));
		}
#endif
	}

	// homogeneous normalized camera coordinates (left)
	const cv::Mat CC_norm_left(K_left.inv() * IC_homo_left);

	for (std::size_t k = 0; k < num_images; ++k)
	{
		const cv::Mat &img_left_before = input_images_left[k];
		const cv::Mat &img_right_before = input_images_right[k];

		// camera coordinates (left)
		cv::Mat CC_left;
		{
			cv::Mat tmp;
#if 0
			// 0 0 0 ...   0 1 1 1 ...   1 ... 639 639 639 ... 639
			// 0 1 2 ... 479 0 1 2 ... 479 ...   0   1   2 ... 479

			((cv::Mat)img_left_before.t()).convertTo(tmp, CV_64FC1, 1.0, 0.0);
#else
			// 0 1 2 ... 639 0 1 2 ... 639 ...   0   1   2 ... 639
			// 0 0 0 ...   0 1 1 1 ...   1 ... 479 479 479 ... 479

			img_left_before.convertTo(tmp, CV_64FC1, 1.0, 0.0);
#endif
			cv::repeat(tmp.reshape(1, 1), 3, 1, CC_left);
			CC_left = CC_left.mul(CC_norm_left);
		}

		// camera coordinates (right)
		cv::Mat CC_right;
#if 0
		cv::repeat(T, 1, imageSize_left.width*imageSize_left.height, CC_right);
		CC_right = R.t() * (CC_left - CC_right);
#else
		cv::repeat(T, 1, imageSize_left.width*imageSize_left.height, CC_right);
		CC_right = R * CC_left + CC_right;
#endif

		// homogeneous normalized camera coordinates (right)
		cv::Mat CC_norm_right;
		cv::repeat(CC_right(cv::Range(2, 3), cv::Range::all()), 3, 1, CC_norm_right);
		CC_norm_right = CC_right / CC_norm_right;

		// homogeneous image coordinates (right)
		const cv::Mat IC_homo_right(K_right * CC_norm_right);  // zero-based coordinates

		// the left image is mapped onto the right image.
		cv::Mat img_left_mapped(img_right_before.size(), img_left_before.type(), cv::Scalar::all(0));
		for (int idx = 0; idx < imageSize_left.height*imageSize_left.width; ++idx)
		{
			const int &cc = (int)cvRound(IC_homo_right.at<double>(0,idx));
			const int &rr = (int)cvRound(IC_homo_right.at<double>(1,idx));
			if (0 <= cc && cc < imageSize_right.width && 0 <= rr && rr < imageSize_right.height)
				img_left_mapped.at<unsigned short>(rr, cc) = (unsigned short)cvRound(CC_left.at<double>(2, idx));
		}

		output_images_left.push_back(img_left_mapped);
		output_images_right.push_back(img_right_before);
	}
}

// RGB (left) to IR (right)
void rectify_kinect_images_from_RGB_to_IR_using_depth(
	const std::vector<cv::Mat> &input_images_left, const std::vector<cv::Mat> &input_images_right, std::vector<cv::Mat> &output_images_left, std::vector<cv::Mat> &output_images_right,
	const cv::Size &imageSize_left, const cv::Size &imageSize_right,
	const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &R, const cv::Mat &T
)
{
	throw std::runtime_error("not yet implemented");
}

void load_kinect_sensor_parameters_from_IR_to_RGB(
	cv::Mat &K_ir, cv::Mat &distCoeffs_ir, cv::Mat &K_rgb, cv::Mat &distCoeffs_rgb,
	cv::Mat &R_ir_to_rgb, cv::Mat &T_ir_to_rgb
)
{
	// [ref]
	//	Camera Calibration Toolbox for Matlab: http://www.vision.caltech.edu/bouguetj/calib_doc/
	//	http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

	// Caution:
	//	In order to use the calibration results from Camera Calibration Toolbox for Matlab in OpenCV,
	//	a parameter for radial distrtortion, kc(5) has to be active, est_dist(5) = 1.

	// IR (left) to RGB (right)
#if 1
	// the 5th distortion parameter, kc(5) is activated.

	const double fc_ir[] = { 5.865281297534211e+02, 5.866623900166177e+02 };  // [pixel]
	const double cc_ir[] = { 3.371860463542209e+02, 2.485298169373497e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.227084070414958e-01, 5.027511830344261e-01, -2.562850607972214e-03, 6.916249031489476e-03, -5.507709925923052e-01 };  // 5x1 vector
	const double kc_ir[] = { -1.227084070414958e-01, 5.027511830344261e-01, -2.562850607972214e-03, 6.916249031489476e-03, -5.507709925923052e-01, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double fc_rgb[] = { 5.248648751941851e+02, 5.268281060449414e+02 };  // [pixel]
	const double cc_rgb[] = { 3.267484107269922e+02, 2.618261807606497e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.796770514235670e-01, -1.112507253647945e+00, 9.265501548915561e-04, 2.428229310663184e-03, 1.744019737212440e+00 };  // 5x1 vector
	const double kc_rgb[] = { 2.796770514235670e-01, -1.112507253647945e+00, 9.265501548915561e-04, 2.428229310663184e-03, 1.744019737212440e+00, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double rotVec[] = { -1.936270295074452e-03, 1.331596538715070e-02, 3.404073398703758e-03 };
	const double transVec[] = { 2.515260082139980e+01, 4.059127243511693e+00, -5.588303932036697e+00 };  // [mm]
#else
	// the 5th distortion parameter, kc(5) is deactivated.

	const double fc_ir[] = { 5.864902565580264e+02, 5.867305900503998e+02 };  // [pixel]
	const double cc_ir[] = { 3.376088045224677e+02, 2.480083390372575e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.123867977947529e-01, 3.552017514491446e-01, -2.823972305243438e-03, 7.246763414437084e-03, 0.0 };  // 5x1 vector
	const double kc_ir[] = { -1.123867977947529e-01, 3.552017514491446e-01, -2.823972305243438e-03, 7.246763414437084e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double fc_rgb[] = { 5.256215953836251e+02, 5.278165866956751e+02 };  // [pixel]
	const double cc_rgb[] = { 3.260532981578608e+02, 2.630788286947369e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.394862387380747e-01, -5.840355691714197e-01, 2.567740590187774e-03, 2.044179978023951e-03, 0.0 };  // 5x1 vector
	const double kc_rgb[] = { 2.394862387380747e-01, -5.840355691714197e-01, 2.567740590187774e-03, 2.044179978023951e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double rotVec[] = { 1.121432126402549e-03, 1.535221550916760e-02, 3.701648572107407e-03 };
	const double transVec[] = { 2.512732389978993e+01, 3.724869927389498e+00, -4.534758982979088e+00 };  // [mm]
#endif

	//
	cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)).copyTo(K_ir);
	K_ir.at<double>(0, 0) = fc_ir[0];
	K_ir.at<double>(0, 1) = alpha_c_ir * fc_ir[0];
	K_ir.at<double>(0, 2) = cc_ir[0];
	K_ir.at<double>(1, 1) = fc_ir[1];
	K_ir.at<double>(1, 2) = cc_ir[1];
	K_ir.at<double>(2, 2) = 1.0;
	cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)).copyTo(K_rgb);
	K_rgb.at<double>(0, 0) = fc_rgb[0];
	K_rgb.at<double>(0, 1) = alpha_c_rgb * fc_rgb[0];
	K_rgb.at<double>(0, 2) = cc_rgb[0];
	K_rgb.at<double>(1, 1) = fc_rgb[1];
	K_rgb.at<double>(1, 2) = cc_rgb[1];
	K_rgb.at<double>(2, 2) = 1.0;

	cv::Mat(8, 1, CV_64FC1, (void *)kc_ir).copyTo(distCoeffs_ir);
	cv::Mat(8, 1, CV_64FC1, (void *)kc_rgb).copyTo(distCoeffs_rgb);

    cv::Rodrigues(cv::Mat(3, 1, CV_64FC1, (void *)rotVec), R_ir_to_rgb);
	cv::Mat(3, 1, CV_64FC1, (void *)transVec).copyTo(T_ir_to_rgb);
}

void load_kinect_sensor_parameters_from_RGB_to_IR(
	cv::Mat &K_rgb, cv::Mat &distCoeffs_rgb, cv::Mat &K_ir, cv::Mat &distCoeffs_ir,
	cv::Mat &R_rgb_to_ir, cv::Mat &T_rgb_to_ir
)
{
	// [ref]
	//	Camera Calibration Toolbox for Matlab: http://www.vision.caltech.edu/bouguetj/calib_doc/
	//	http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

	// Caution:
	//	In order to use the calibration results from Camera Calibration Toolbox for Matlab in OpenCV,
	//	a parameter for radial distrtortion, kc(5) has to be active, est_dist(5) = 1.

	// RGB (left) to IR (right)
#if 1
	// the 5th distortion parameter, kc(5) is activated.

	const double fc_rgb[] = { 5.248648079874888e+02, 5.268280486062615e+02 };  // [pixel]
	const double cc_rgb[] = { 3.267487100838014e+02, 2.618261169946102e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.796764337988712e-01, -1.112497355183840e+00, 9.264749543097661e-04, 2.428507887293728e-03, 1.743975665436613e+00 };  // 5x1 vector
	const double kc_rgb[] = { 2.796764337988712e-01, -1.112497355183840e+00, 9.264749543097661e-04, 2.428507887293728e-03, 1.743975665436613e+00, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double fc_ir[] = { 5.865282023957649e+02, 5.866624209441105e+02 };  // [pixel]
	const double cc_ir[] = { 3.371875014947813e+02, 2.485295493095561e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.227176734054719e-01, 5.028746725848668e-01, -2.563029340202278e-03, 6.916996280663117e-03, -5.512162545452755e-01 };  // 5x1 vector
	const double kc_ir[] = { -1.227176734054719e-01, 5.028746725848668e-01, -2.563029340202278e-03, 6.916996280663117e-03, -5.512162545452755e-01, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double rotVec[] = { 1.935939237060295e-03, -1.331788958930441e-02, -3.404128236480992e-03 };
	const double transVec[] = { -2.515262012891160e+01, -4.059118899373607e+00, 5.588237589014362e+00 };  // [mm]
#else
	// the 5th distortion parameter, kc(5) is deactivated.

	const double fc_rgb[] = { 5.256217798767822e+02, 5.278167798992870e+02 };  // [pixel]
	const double cc_rgb[] = { 3.260534767468189e+02, 2.630800669346188e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.394861400525463e-01, -5.840298777969020e-01, 2.568959896208732e-03, 2.044336479083819e-03, 0.0 };  // 5x1 vector
	const double kc_rgb[] = { 2.394861400525463e-01, -5.840298777969020e-01, 2.568959896208732e-03, 2.044336479083819e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double fc_ir[] = { 5.864904832545356e+02, 5.867308191567271e+02 };  // [pixel]
	const double cc_ir[] = { 3.376079004969836e+02, 2.480098376453992e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.123902857373373e-01, 3.552211727724343e-01, -2.823183218548772e-03, 7.246270574438420e-03, 0.0 };  // 5x1 vector
	const double kc_ir[] = { -1.123902857373373e-01, 3.552211727724343e-01, -2.823183218548772e-03, 7.246270574438420e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double rotVec[] = { -1.121214964017936e-03, -1.535031632771925e-02, -3.701579055761772e-03 };
	const double transVec[] = { -2.512730902761022e+01, -3.724884753207001e+00, 4.534776794502955e+00 };  // [mm]
#endif

	//
	cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)).copyTo(K_rgb);
	K_rgb.at<double>(0, 0) = fc_rgb[0];
	K_rgb.at<double>(0, 1) = alpha_c_rgb * fc_rgb[0];
	K_rgb.at<double>(0, 2) = cc_rgb[0];
	K_rgb.at<double>(1, 1) = fc_rgb[1];
	K_rgb.at<double>(1, 2) = cc_rgb[1];
	K_rgb.at<double>(2, 2) = 1.0;
	cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)).copyTo(K_ir);
	K_ir.at<double>(0, 0) = fc_ir[0];
	K_ir.at<double>(0, 1) = alpha_c_ir * fc_ir[0];
	K_ir.at<double>(0, 2) = cc_ir[0];
	K_ir.at<double>(1, 1) = fc_ir[1];
	K_ir.at<double>(1, 2) = cc_ir[1];
	K_ir.at<double>(2, 2) = 1.0;

	cv::Mat(8, 1, CV_64FC1, (void *)kc_rgb).copyTo(distCoeffs_rgb);
	cv::Mat(8, 1, CV_64FC1, (void *)kc_ir).copyTo(distCoeffs_ir);

    cv::Rodrigues(cv::Mat(3, 1, CV_64FC1, (void *)rotVec), R_rgb_to_ir);
	cv::Mat(3, 1, CV_64FC1, (void *)transVec).copyTo(T_rgb_to_ir);
}

void rectify_kinect_images_using_opencv(
	const bool use_IR_to_RGB, const std::size_t &num_images, const cv::Size &imageSize_ir, const cv::Size &imageSize_rgb,
	const std::vector<cv::Mat> &ir_input_images, const std::vector<cv::Mat> &rgb_input_images, std::vector<cv::Mat> &ir_output_images, std::vector<cv::Mat> &rgb_output_images,
	const cv::Mat &K_ir, const cv::Mat &K_rgb, const cv::Mat &distCoeffs_ir, const cv::Mat &distCoeffs_rgb, const cv::Mat &R, const cv::Mat &T
)
{
	if (use_IR_to_RGB)
		rectify_images_using_opencv(
			ir_input_images, rgb_input_images, ir_output_images, rgb_output_images,
			imageSize_ir, imageSize_rgb,
			K_ir, K_rgb, distCoeffs_ir, distCoeffs_rgb, R, T
		);
	else
		rectify_images_using_opencv(
			rgb_input_images, ir_input_images, rgb_output_images, ir_output_images,
			imageSize_rgb, imageSize_ir,
			K_rgb, K_ir, distCoeffs_rgb, distCoeffs_ir, R, T
		);
}

void rectify_kinect_images_using_depth(
	const bool use_IR_to_RGB, const std::size_t &num_images, const cv::Size &imageSize_ir, const cv::Size &imageSize_rgb,
	const std::vector<cv::Mat> &ir_input_images, const std::vector<cv::Mat> &rgb_input_images, std::vector<cv::Mat> &ir_output_images, std::vector<cv::Mat> &rgb_output_images,
	const cv::Mat &K_ir, const cv::Mat &K_rgb, const cv::Mat &distCoeffs_ir, const cv::Mat &distCoeffs_rgb, const cv::Mat &R, const cv::Mat &T
)
{
	// undistort images
	// TODO [check] >> is undistortion required before rectification?
	//	Undistortion process is required before rectification,
	//	since currently image undistortion is not applied during rectification process in rectify_kinect_images_from_IR_to_RGB_using_depth() & rectify_kinect_images_from_RGB_to_IR_using_depth().
#if 1
	std::vector<cv::Mat> ir_input_images2, rgb_input_images2;
	{
		ir_input_images2.reserve(num_images);
		rgb_input_images2.reserve(num_images);

#if 0
		undistort_images_using_opencv(ir_input_images, ir_input_images2, imageSize_ir, K_ir, distCoeffs_ir);
		undistort_images_using_opencv(rgb_input_images, rgb_input_images2, imageSize_rgb, K_rgb, distCoeffs_rgb);
#elif 1
		undistort_images_using_formula<unsigned short>(ir_input_images, ir_input_images2, imageSize_ir, K_ir, distCoeffs_ir);
		undistort_images_using_formula<cv::Vec3b>(rgb_input_images, rgb_input_images2, imageSize_rgb, K_rgb, distCoeffs_rgb);
#else
		undistort_images_using_formula<unsigned short>(ir_input_images, ir_input_images2, imageSize_ir, K_ir, distCoeffs_ir);

		std::vector<cv::Mat> rgb_input_gray_images;
		rgb_input_gray_images.reserve(num_images);
		for (std::vector<cv::Mat>::const_iterator cit = rgb_input_images.begin(); cit != rgb_input_images.end(); ++cit)
		{
			cv::Mat gray;
			cv::cvtColor(*cit, gray, CV_BGR2GRAY);
			rgb_input_gray_images.push_back(gray);
		}

		undistort_images_using_formula<unsigned char>(rgb_input_gray_images, rgb_input_images2, imageSize_rgb, K_rgb, distCoeffs_rgb);
#endif
	}
#else
	const std::vector<cv::Mat> ir_input_images2(ir_input_images.begin(), ir_input_images.end()), rgb_input_images2(rgb_input_images.begin(), rgb_input_images.end());
#endif

	// rectify images
	if (use_IR_to_RGB)
		rectify_kinect_images_from_IR_to_RGB_using_depth(
			ir_input_images2, rgb_input_images2, ir_output_images, rgb_output_images,
			imageSize_ir, imageSize_rgb,
			K_ir, K_rgb, R, T
		);
	else
		rectify_kinect_images_from_RGB_to_IR_using_depth(
			rgb_input_images2, ir_input_images2, rgb_output_images, ir_output_images,
			imageSize_rgb, imageSize_ir,
			K_rgb, K_ir, R, T
		);  // not yet implemented.
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_rectification()
{
	// [ref] stereo_camera_calibration() in opencv_stereo_camera_calibration.cpp
#if 1
	// [ref] http://blog.martinperis.com/2011/01/opencv-stereo-camera-calibration.html
	const std::string imagelistfn("./data/machine_vision/opencv/camera_calibration/stereo_calib_2.xml");

	const cv::Size imageSize_left(640, 480), imageSize_right(640, 480);
	const cv::Size boardSize(9, 6);
	const float squareSize = 2.5f;  // Set this to your actual square size, [cm]
	
	//
	std::vector<std::string> imageList;
	const bool ok = local::readStringList(imagelistfn, imageList);
	if (!ok || imageList.empty())
	{
		std::cout << "can not open " << imagelistfn << " or the std::string list is empty" << std::endl;
		return;
	}

	const std::size_t num_images = (int)imageList.size() / 2;

	std::vector<cv::Mat> input_images_left, input_images_right;
	input_images_left.reserve(num_images);
	input_images_right.reserve(num_images);
	bool isLeft = true;
	for (std::vector<std::string>::const_iterator cit = imageList.begin(); cit != imageList.end(); ++cit)
	{
		if (isLeft)
			input_images_left.push_back(cv::imread(*cit, CV_LOAD_IMAGE_COLOR));
		else
			input_images_right.push_back(cv::imread(*cit, CV_LOAD_IMAGE_COLOR));

		isLeft = !isLeft;
	}

	//
	const double matK_left[] = {  // row-major
		5.9858962993336570e+002, 0., 2.8896200893303171e+002,
		0., 5.9858962993336570e+002, 2.2723599894016587e+002,
		0., 0., 1.
	};
	const double matK_right[] = {  // row-major
		5.9858962993336570e+002, 0., 3.3168905477411886e+002,
		0., 5.9858962993336570e+002, 2.4010300779442360e+002,
		0., 0., 1.
	};
	const double vecDistCoeffs_left[] = { -1.2061354444336511e-001, -1.9304020915488919e-002, 0., 0., 0., 0., 0., -1.8530356850628668e-001 };  // 8x1 vector
	const double vecDistCoeffs_right[] = { -1.2592386298383618e-001, -7.4786119168434750e-002, 0., 0., 0., 0., 0., -3.7945350098585728e-001 };  // 8x1 vector
	const double matR[] = {  // row-major
		9.9997336147015936e-001, -1.7315598556007967e-003, -7.0907016956375793e-003,
		1.7671899978048300e-003, 9.9998582952301485e-001, 5.0217320398037758e-003,
		7.0819057874066957e-003, -5.0341288853589135e-003, 9.9996225136591232e-001
	};
	const double vecT[] = { -7.8728599512360793e+000, -6.5575195421650509e-002, 1.9230377565016391e-002 };
	const double matR_left[] = {  // row-major
		9.9993311414905961e-001, 6.6096420106741610e-003, -9.4910410732816225e-003,
		-6.5856100280237062e-003, 9.9997503495514062e-001, 2.5610948877125301e-003,
		9.5077320493786172e-003, -2.4984192914330339e-003, 9.9995167929871565e-001
	};
	const double matR_right[] = {  // row-major
		9.9996233055122574e-001, 8.3289586816397941e-003, -2.4425244811158747e-003,
		-8.3351111840451013e-003, 9.9996208794000574e-001, -2.5196436679870682e-003,
		2.4214458719780925e-003, 2.5399074675188441e-003, 9.9999384271601677e-001
	};
	const double matP_left[] = {  // row-major
		5.3952323419480547e+002, 0., 3.1448822021484375e+002, 0.,
		0., 5.3952323419480547e+002, 2.3376013755798340e+002, 0.,
		0., 0., 1., 0.
	};
	const double matP_right[] = {  // row-major
		5.3952323419480547e+002, 0., 3.1448822021484375e+002, -4.2477508736875907e+003,
		0., 5.3952323419480547e+002, 2.3376013755798340e+002, 0.,
		0., 0., 1., 0.
	};
	const double matQ[] = {  // row-major
		1., 0., 0., -3.1448822021484375e+002,
		0., 1., 0., -2.3376013755798340e+002,
		0., 0., 0., 5.3952323419480547e+002,
		0., 0., 1.2701385986095515e-001, 0.
	};
#elif 0
	// Caution: not correctly working

	// Kinect IR & RGB images
	//const std::string imagelistfn("./data/machine_vision/opencv/camera_calibration/stereo_calib_3.xml");
	const std::string imagelistfn("./data/machine_vision/opencv/camera_calibration/stereo_calib_4.xml");

	const cv::Size imageSize_left(640, 480), imageSize_right(640, 480);
	const cv::Size boardSize(7, 5);
	const float squareSize = 10.0f;  // Set this to your actual square size, [cm]

	//
	const double matK_left[] = {};  // row-major
	const double matK_right[] = {};  // row-major
	const double vecDistCoeffs_left[] = {};
	const double vecDistCoeffs_right[] = {};
	const double matR[] = {};  // row-major
	const double vecT[] = {};
	const double matR_left[] = {};  // row-major
	const double matR_right[] = {};  // row-major
	const double matP_left[] = {};  // row-major
	const double matP_right[] = {};  // row-major
	const double matQ[] = {};  // row-major
#else
	// [ref] ${OPENCV_HOME}/samples/cpp/stereo_calib.xml
	const std::string imagelistfn("./data/machine_vision/opencv/camera_calibration/stereo_calib.xml");

	const cv::Size imageSize_left(640, 480), imageSize_right(640, 480);
	const cv::Size boardSize(9, 6);
	const float squareSize = 1.f;  // Set this to your actual square size, [cm]

	//
	std::vector<std::string> imageList;
	const bool ok = local::readStringList(imagelistfn, imageList);
	if (!ok || imageList.empty())
	{
		std::cout << "can not open " << imagelistfn << " or the std::string list is empty" << std::endl;
		return;
	}

	const std::size_t num_images = (int)imageList.size() / 2;

	std::vector<cv::Mat> input_images_left, input_images_right;
	input_images_left.reserve(num_images);
	input_images_right.reserve(num_images);
	{
		bool isLeft = true;
		for (std::vector<std::string>::const_iterator cit = imageList.begin(); cit != imageList.end(); ++cit)
		{
			if (isLeft)
				input_images_left.push_back(cv::imread(*cit, CV_LOAD_IMAGE_GRAYSCALE));
			else
				input_images_right.push_back(cv::imread(*cit, CV_LOAD_IMAGE_GRAYSCALE));

			isLeft = !isLeft;
		}
	}

	//
	const double matK_left[] = {  // row-major
		5.3471311032432391e+002, 0., 3.3513838135674729e+002,
		0., 5.3471311032432391e+002, 2.4020578137651339e+002,
		0., 0., 1.
	};
	const double matK_right[] = {  // row-major
		5.3471311032432391e+002, 0., 3.3401518911545526e+002,
		0., 5.3471311032432391e+002, 2.4159041667844363e+002,
		0., 0., 1.
	};
	const double vecDistCoeffs_left[] = { -2.7456815913629584e-001, -1.8329019064968352e-002, 0., 0., 0., 0., 0., -2.4481064038800596e-001 };  // 8x1 vector
	const double vecDistCoeffs_right[] = { -2.8073450162365265e-001, 9.3000165783150540e-002, 0., 0., 0., 0., 0., 1.6314434959666235e-002 };  // 8x1 vector
	const double matR[] = {  // row-major
		9.9975845371004723e-001, 5.2938494283308168e-003, -2.1330949194199155e-002,
		-4.9128856780201770e-003, 9.9982820089904900e-001, 1.7872667436219555e-002,
		2.1421899766595125e-002, -1.7763553844914036e-002, 9.9961270418356973e-001
	};
	const double vecT[] = { -3.3385325916025854e+000, 4.8752483611574089e-002, -1.0621381929002099e-001 };
	const double matR_left[] = {  // row-major
		9.9989926797130868e-001, -9.8657055577613794e-003, 1.0204007266193976e-002,
		9.7747996340919619e-003, 9.9991243357170068e-001, 8.9206771510116165e-003,
		-1.0291122511871056e-002, -8.8200364266130208e-003,	9.9990814565882946e-001
	};
	const double matR_right[] = {  // row-major
		9.9938785729250423e-001, -1.4594028603110288e-002, 3.1795047183937220e-002,
		1.4875537921264092e-002, 9.9985206333748144e-001, -8.6353813658799159e-003,
		-3.1664318528119781e-002, 9.1030637102433969e-003, 9.9945710521424436e-001
	};
	const double matP_left[] = {  // row-major
		4.2656433611135799e+002, 0., 3.2185688400268555e+002, 0.,
		0., 4.2656433611135799e+002, 2.4122886657714844e+002, 0.,
		0., 0., 1., 0.
	};
	const double matP_right[] = {  // row-major
		4.2656433611135799e+002, 0., 3.2185688400268555e+002, -1.4249712242664141e+003,
		0., 4.2656433611135799e+002, 2.4122886657714844e+002, 0.,
		0., 0., 1., 0.
	};
	const double matQ[] = {  // row-major
		1., 0., 0., -3.2185688400268555e+002,
		0., 1., 0., -2.4122886657714844e+002,
		0., 0., 0., 4.2656433611135799e+002,
		0., 0., 2.9934943867442410e-001, 0.
	};
#endif

	const cv::Mat K_left(3, 3, CV_64FC1, (void *)matK_left), K_right(3, 3, CV_64FC1, (void *)matK_right);  // row-major
	const cv::Mat distCoeffs_left(8, 1, CV_64FC1, (void *)vecDistCoeffs_left), distCoeffs_right(8, 1, CV_64FC1, (void *)vecDistCoeffs_right);
	const cv::Mat R(3, 3, CV_64FC1, (void *)matR);  // row-major
	const cv::Mat T(3, 1, CV_64FC1, (void *)vecT);
	const cv::Mat R_left(3, 3, CV_64FC1, (void *)matR_left), R_right(3, 3, CV_64FC1, (void *)matR_right);  // row-major
	const cv::Mat P_left(3, 4, CV_64FC1, (void *)matP_left), P_right(3, 4, CV_64FC1, (void *)matP_right);  // row-major
	//const cv::Mat Q(4, 4, CV_64FC1, (void *)matQ);  // row-major

	// OpenCV can handle left-right or up-down camera arrangements
	//const bool isVerticalStereo = std::fabs(P_right.at<double>(1, 3)) > std::fabs(P_right.at<double>(0, 3));

	std::vector<cv::Mat> output_images_left, output_images_right;
	output_images_left.reserve(num_images);
	output_images_right.reserve(num_images);
	{
		const int64 start = cv::getTickCount();

		// precompute maps for cv::remap()
		cv::Mat rmap_left[2], rmap_right[2];
		cv::initUndistortRectifyMap(K_left, distCoeffs_left, R_left, P_left, imageSize_left, CV_16SC2, rmap_left[0], rmap_left[1]);
		cv::initUndistortRectifyMap(K_right, distCoeffs_right, R_right, P_right, imageSize_right, CV_16SC2, rmap_right[0], rmap_right[1]);

		// rectify images
		for (std::size_t i = 0; i < num_images; ++i)
		{
			cv::Mat rimg_left, rimg_right;
			cv::remap(input_images_left[i], rimg_left, rmap_left[0], rmap_left[1], CV_INTER_LINEAR);
			cv::remap(input_images_right[i], rimg_right, rmap_right[0], rmap_right[1], CV_INTER_LINEAR);

			output_images_left.push_back(rimg_left);
			output_images_right.push_back(rimg_right);
		}

		const int64 elapsed = cv::getTickCount() - start;
		const double freq = cv::getTickFrequency();
		const double etime = elapsed * 1000.0 / freq;
		const double fps = freq / elapsed;
		std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;
	}

	// show results
	//cv::Mat img_left_after2;
	//double minVal = 0.0, maxVal = 0.0;
	for (std::size_t k = 0; k < num_images; ++k)
	{
		const cv::Mat &img_left_after = output_images_left[k];
		const cv::Mat &img_right_after = output_images_right[k];

		//
		//cv::minMaxLoc(img_left_after, &minVal, &maxVal);
		//img_left_after.convertTo(img_left_after2, CV_32FC1, 1.0 / maxVal, 0.0);

		cv::imshow("rectified left image", img_left_after);
		cv::imshow("rectified right image", img_right_after);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

void kinect_image_rectification()
{
	// more integrated implementation
	//	[ref] ${SWL_CPP_HOME}/app/swl_kinect_segmentation_app/KinectSensor.h & cpp

	const bool use_IR_to_RGB = true;

	// load the camera parameters of a Kinect sensor
	cv::Mat K_ir, K_rgb;
	cv::Mat distCoeffs_ir, distCoeffs_rgb;
	cv::Mat R, T;
	if (use_IR_to_RGB)
		local::load_kinect_sensor_parameters_from_IR_to_RGB(K_ir, distCoeffs_ir, K_rgb, distCoeffs_rgb, R, T);
	else
		local::load_kinect_sensor_parameters_from_RGB_to_IR(K_rgb, distCoeffs_rgb, K_ir, distCoeffs_ir, R, T);

	// image rectification
	{
		// prepare input images
		const std::size_t num_images = 4;
		const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480);

		std::vector<std::string> ir_image_filenames;
		ir_image_filenames.reserve(num_images);
		ir_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_depth_20130530T103805.png");
		ir_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_depth_20130531T023152.png");
		ir_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_depth_20130531T023346.png");
		ir_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_depth_20130531T023359.png");

		std::vector<std::string> rgb_image_filenames;
		rgb_image_filenames.reserve(num_images);
		rgb_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_rgba_20130530T103805.png");
		rgb_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_rgba_20130531T023152.png");
		rgb_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_rgba_20130531T023346.png");
		rgb_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_rgba_20130531T023359.png");

		std::vector<cv::Mat> ir_input_images, rgb_input_images;
		ir_input_images.reserve(num_images);
		rgb_input_images.reserve(num_images);
		for (std::size_t k = 0; k < num_images; ++k)
		{
			ir_input_images.push_back(cv::imread(ir_image_filenames[k], CV_LOAD_IMAGE_UNCHANGED));
			rgb_input_images.push_back(cv::imread(rgb_image_filenames[k], CV_LOAD_IMAGE_COLOR));
		}

		// rectify images
		std::vector<cv::Mat> ir_output_images, rgb_output_images;
		ir_output_images.reserve(num_images);
		rgb_output_images.reserve(num_images);

		{
			const int64 start = cv::getTickCount();

#if 0
			local::rectify_kinect_images_using_opencv(use_IR_to_RGB, num_images, imageSize_ir, imageSize_rgb, ir_input_images, rgb_input_images, ir_output_images, rgb_output_images, K_ir, K_rgb, distCoeffs_ir, distCoeffs_rgb, R, T);  // using OpenCV
#else
			local::rectify_kinect_images_using_depth(use_IR_to_RGB, num_images, imageSize_ir, imageSize_rgb, ir_input_images, rgb_input_images, ir_output_images, rgb_output_images, K_ir, K_rgb, distCoeffs_ir, distCoeffs_rgb, R, T);  // using Kinect's depth information
#endif

			const int64 elapsed = cv::getTickCount() - start;
			const double freq = cv::getTickFrequency();
			const double etime = elapsed * 1000.0 / freq;
			const double fps = freq / elapsed;
			std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;
		}

		// show results
		cv::Mat ir_img_after2;
		double minVal = 0.0, maxVal = 0.0;
		for (std::size_t k = 0; k < num_images; ++k)
		{
			const cv::Mat &ir_img_after = ir_output_images[k];
			const cv::Mat &rgb_img_after = rgb_output_images[k];

			//
			cv::minMaxLoc(ir_img_after, &minVal, &maxVal);
			ir_img_after.convertTo(ir_img_after2, CV_32FC1, 1.0 / maxVal, 0.0);

			cv::imshow(use_IR_to_RGB ? "rectified depth (left) image" : "rectified depth (right) image", ir_img_after2);
			cv::imshow(use_IR_to_RGB ? "rectified RGB (right) image" : "rectified RGB (left) image", rgb_img_after);

			const unsigned char key = cv::waitKey(0);
			if (27 == key)
				break;
		}

#if 0
		// save results
		for (std::size_t k = 0; k < num_images; ++k)
		{
			std::ostringstream strm1, strm2;
			strm1 << "./data/machine_vision/opencv/image_undistortion/rectified_image_depth_" << k << ".png";
			cv::imwrite(strm1.str(), ir_output_images[k]);
			strm2 << "./data/machine_vision/opencv/image_undistortion/rectified_image_rgb_" << k << ".png";
			cv::imwrite(strm2.str(), rgb_output_images[k]);
		}
#endif

		cv::destroyAllWindows();
	}
}

}  // namespace my_opencv
