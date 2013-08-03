//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>


#define __USE_OPENCV_REMAP 1

namespace {
namespace local {

// [ref] readStringList() in opencv_camera_calibration.cop
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

// undistort images
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
					(1.0 - alpha_rr) * (1.0 - alpha_cc) * img_before.at<T>(rr_0, cc_0) +
					(1.0 - alpha_rr) * alpha_cc * img_before.at<T>(rr_0, cc_1) +
					alpha_rr * (1.0 - alpha_cc) * img_before.at<T>(rr_1, cc_0) +
					alpha_rr * alpha_cc * img_before.at<T>(rr_1, cc_1);
			}
#endif
		}

		output_images.push_back(img_after);
	}
}

void load_kinect_sensor_parameters(cv::Mat &K_ir, cv::Mat &distCoeffs_ir, cv::Mat &K_rgb, cv::Mat &distCoeffs_rgb)
{
	// [ref]
	//	Camera Calibration Toolbox for Matlab: http://www.vision.caltech.edu/bouguetj/calib_doc/
	//	http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

	//	In order to use the calibration results from Camera Calibration Toolbox for Matlab,
	//	a parameter for radial distrtortion, kc(5) has to be active, est_dist(5) = 1.

#if 1
	// the 5th distortion parameter, kc(5) is activated.

	// IR
	const double fc_ir[] = { 5.857251103301124e+02, 5.861509849627823e+02 };  // [pixel]
	const double cc_ir[] = { 3.360396440069350e+02, 2.468430078952277e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.113144398698150e-01, 3.902042354943196e-01, -2.473313414949828e-03, 6.053929513996014e-03, -2.342535197486739e-01 };  // 5x1 vector
	const double kc_ir[] = { -1.113144398698150e-01, 3.902042354943196e-01, -2.473313414949828e-03, 6.053929513996014e-03, -2.342535197486739e-01, 0.0, 0.0, 0.0 };  // 8x1 vector

	// RGB
	const double fc_rgb[] = { 5.261769128081118e+02, 5.280693668967953e+02 };  // [pixel]
	const double cc_rgb[] = { 3.290215649965892e+02, 2.651462857334770e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.639717236885097e-01, -9.026376922133396e-01, 2.569103898876239e-03, 4.773654687023216e-03, 1.074728662132601e+00 };  // 5x1 vector
	const double kc_rgb[] = { 2.639717236885097e-01, -9.026376922133396e-01, 2.569103898876239e-03, 4.773654687023216e-03, 1.074728662132601e+00, 0.0, 0.0, 0.0 };  // 8x1 vector
#else
	// the 5th distortion parameter, kc(5) is deactivated.

	// IR
	const double fc_ir[] = { 5.857535922475207e+02, 5.865708030703412e+02 };  // [pixel]
	const double cc_ir[] = { 3.351932174524685e+02, 2.464165684432059e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.063901580499479e-01, 3.395192881812036e-01, -2.211031053332312e-03, 5.882227715342140e-03, 0.0 };  // 5x1 vector
	const double kc_ir[] = { -1.063901580499479e-01, 3.395192881812036e-01, -2.211031053332312e-03, 5.882227715342140e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector

	// RGB
	const double fc_rgb[] = { 5.266814231294437e+02, 5.280641466171643e+02 };  // [pixel]
	const double cc_rgb[] = { 3.276528954184697e+02, 2.652059636854492e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.322255151854028e-01, -5.598137839760616e-01, 2.277053552942137e-03, 3.720963676783346e-03, 0.0 };  // 5x1 vector
	const double kc_rgb[] = { 2.322255151854028e-01, -5.598137839760616e-01, 2.277053552942137e-03, 3.720963676783346e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector
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
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_undistortion()
{
	// [ref] camera_calibration() in opencv_camera_calibration.cop
#if 0
	// [ref] ${OPENCV_HOME}/samples/cpp/stereo_calib.xml
	const std::string inputFilename("./data/machine_vision/opencv/camera_calibration/camera_calib.xml");

	const cv::Size imageSize(640, 480);

	//
	std::vector<std::string> imageList;
	if (!local::readStringList(inputFilename, imageList))
	{
		std::cerr << "input file list cannot load" << std::endl;
		return;
	}

	std::vector<cv::Mat> input_images;
	input_images.reserve(imageList.size());
	for (std::vector<std::string>::const_iterator cit = imageList.begin(); cit != imageList.end(); ++cit)
		input_images.push_back(cv::imread(*cit, CV_LOAD_IMAGE_GRAYSCALE));

	//
	const double matK[] = { 5.3587897712223048e+002, 0., 3.4227646116834137e+002, 0., 5.3583540725808348e+002, 2.3552854434700765e+002, 0., 0., 1. };  // row-major
	//const double vecDistCoeffs[] = { -2.6621060985333589e-001, -3.9867063477248534e-002, 1.7924689671144441e-003, -2.9304640993473073e-004, 2.4053846993400707e-001 };  // 5x1 vector
	const double vecDistCoeffs[] = { -2.6621060985333589e-001, -3.9867063477248534e-002, 1.7924689671144441e-003, -2.9304640993473073e-004, 2.4053846993400707e-001, 0.0, 0.0, 0.0 };  // 8x1 vector
#elif 0
	// [ref] http://blog.martinperis.com/2011/01/opencv-stereo-camera-calibration.html
	const std::string inputFilename("./data/machine_vision/opencv/camera_calibration/camera_calib_2.xml");

	const cv::Size imageSize(640, 480);

	//
	std::vector<std::string> imageList;
	if (!local::readStringList(inputFilename, imageList))
	{
		std::cerr << "input file list cannot load" << std::endl;
		return;
	}

	std::vector<cv::Mat> input_images;
	input_images.reserve(imageList.size());
	for (std::vector<std::string>::const_iterator cit = imageList.begin(); cit != imageList.end(); ++cit)
		input_images.push_back(cv::imread(*cit, CV_LOAD_IMAGE_COLOR));

	//
	const double matK[] = { 5.8574839894570380e+002, 0., 2.9515016167159104e+002, 0., 5.8831673934047501e+002, 2.1618663678777577e+002, 0., 0., 1. };  // row-major
	//const double vecDistCoeffs[] = { -1.2075418632920232e-001, -4.6789760656275450e-002, -3.4245413462073268e-003, 3.1902395837533786e-003, 2.3241583016410450e-001 };  // 5x1 vector
	const double vecDistCoeffs[] = { -1.2075418632920232e-001, -4.6789760656275450e-002, -3.4245413462073268e-003, 3.1902395837533786e-003, 2.3241583016410450e-001, 0.0, 0.0, 0.0 };  // 8x1 vector
#elif 1
	// Kinect RGB images
	const std::string inputFilename("./data/machine_vision/opencv/camera_calibration/camera_calib_3.xml");

	const cv::Size imageSize(640, 480);

	//
	std::vector<std::string> imageList;
	if (!local::readStringList(inputFilename, imageList))
	{
		std::cerr << "input file list cannot load" << std::endl;
		return;
	}

	std::vector<cv::Mat> input_images;
	input_images.reserve(imageList.size());
	for (std::vector<std::string>::const_iterator cit = imageList.begin(); cit != imageList.end(); ++cit)
		input_images.push_back(cv::imread(*cit, CV_LOAD_IMAGE_COLOR));

	//
	const double matK[] = { 5.2635228817969698e+002, 0., 3.2721575024118914e+002, 0., 5.2765917576898983e+002, 2.6550336208783216e+002, 0., 0., 1. };  // row-major
	//const double vecDistCoeffs[] = { 2.5703815534648017e-001, -8.6596989999336349e-001, 2.2803193915667845e-003, 3.3064737839550973e-003, 9.9706986903207828e-001 };  // 5x1 vector
	const double vecDistCoeffs[] = { 2.5703815534648017e-001, -8.6596989999336349e-001, 2.2803193915667845e-003, 3.3064737839550973e-003, 9.9706986903207828e-001, 0.0, 0.0, 0.0 };  // 8x1 vector
#endif

	const cv::Mat K(3, 3, CV_64FC1, (void *)matK);  // row-major
	const cv::Mat distCoeffs(8, 1, CV_64FC1, (void *)vecDistCoeffs);

	//
	std::vector<cv::Mat> output_images;
	output_images.reserve(input_images.size());
	{
		const int64 start = cv::getTickCount();

		local::undistort_images_using_opencv(input_images, output_images, imageSize, K, distCoeffs);

		const int64 elapsed = cv::getTickCount() - start;
		const double freq = cv::getTickFrequency();
		const double etime = elapsed * 1000.0 / freq;
		const double fps = freq / elapsed;
		std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;
	}

	// show results
	cv::Mat img_after;
	double minVal = 0.0, maxVal = 0.0;
	for (std::vector<cv::Mat>::const_iterator cit = output_images.begin(); cit != output_images.end(); ++cit)
	{
		cv::minMaxLoc(*cit, &minVal, &maxVal);
		cit->convertTo(img_after, CV_32FC1, 1.0 / maxVal, 0.0);

		cv::imshow("undistorted left image", img_after);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

void kinect_image_undistortion()
{
	// load the camera parameters of a Kinect sensor.
	cv::Mat K_ir, K_rgb;
	cv::Mat distCoeffs_ir, distCoeffs_rgb;
	local::load_kinect_sensor_parameters(K_ir, distCoeffs_ir, K_rgb, distCoeffs_rgb);

	// undistort IR images
	{
		// prepare input images
		std::vector<std::string> input_image_filenames;
		input_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_depth_20130530T103805.png");
		input_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_depth_20130531T023152.png");
		input_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_depth_20130531T023346.png");
		input_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_depth_20130531T023359.png");

		const std::size_t num_images = input_image_filenames.size();
		const cv::Size imageSize(640, 480);

		std::vector<cv::Mat> input_images;
		input_images.reserve(num_images);
		for (std::size_t k = 0; k < num_images; ++k)
			input_images.push_back(cv::imread(input_image_filenames[k], CV_LOAD_IMAGE_UNCHANGED));

		//
		std::vector<cv::Mat> output_images;
		output_images.reserve(num_images);
		{
			const int64 start = cv::getTickCount();

#if 1
			local::undistort_images_using_opencv(input_images, output_images, imageSize, K_ir, distCoeffs_ir);
#else
			local::undistort_images_using_formula<unsigned short>(input_images, output_images, imageSize, K_ir, distCoeffs_ir);
#endif

			const int64 elapsed = cv::getTickCount() - start;
			const double freq = cv::getTickFrequency();
			const double etime = elapsed * 1000.0 / freq;
			const double fps = freq / elapsed;
			std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;
		}

		// show results
		cv::Mat img_after;
		double minVal = 0.0, maxVal = 0.0;
		for (std::vector<cv::Mat>::const_iterator cit = output_images.begin(); cit != output_images.end(); ++cit)
		{
			cv::minMaxLoc(*cit, &minVal, &maxVal);
			cit->convertTo(img_after, CV_32FC1, 1.0 / maxVal, 0.0);

			cv::imshow("undistorted IR image", img_after);

			const unsigned char key = cv::waitKey(0);
			if (27 == key)
				break;
		}

#if 0
		// save results
		for (std::size_t k = 0; k < num_images; ++k)
		{
			std::ostringstream strm;
			strm << "./data/machine_vision/opencv/image_undistortion/undistorted_image_" << k++ << ".png";
			cv::imwrite(strm.str(), output_images[k]);
		}
	#endif
	}

	// undistort RGB images
	{
		// prepare input images
		std::vector<std::string> input_image_filenames;
		input_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_rgba_20130530T103805.png");
		input_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_rgba_20130531T023152.png");
		input_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_rgba_20130531T023346.png");
		input_image_filenames.push_back("./data/machine_vision/opencv/image_undistortion/kinect_rgba_20130531T023359.png");

		const std::size_t num_images = input_image_filenames.size();
		const cv::Size imageSize(640, 480);

		std::vector<cv::Mat> input_images;
		input_images.reserve(num_images);
		for (std::size_t k = 0; k < num_images; ++k)
			input_images.push_back(cv::imread(input_image_filenames[k], CV_LOAD_IMAGE_COLOR));

		//
		std::vector<cv::Mat> output_images;
		output_images.reserve(num_images);
		{
			const int64 start = cv::getTickCount();

#if 0
			local::undistort_images_using_opencv(input_images, output_images, imageSize, K_rgb, distCoeffs_rgb);
#elif 1
			local::undistort_images_using_formula<cv::Vec3b>(input_images, output_images, imageSize, K_rgb, distCoeffs_rgb);
#else
			std::vector<cv::Mat> input_gray_images;
			input_gray_images.reserve(num_images);
			for (std::vector<cv::Mat>::const_iterator cit = input_images.begin(); cit != input_images.end(); ++cit)
			{
				cv::Mat gray;
				cv::cvtColor(*cit, gray, CV_BGR2GRAY);
				input_gray_images.push_back(gray);
			}

			local::undistort_images_using_formula<unsigned char>(input_gray_images, output_images, imageSize, K_rgb, distCoeffs_rgb);
#endif

			const int64 elapsed = cv::getTickCount() - start;
			const double freq = cv::getTickFrequency();
			const double etime = elapsed * 1000.0 / freq;
			const double fps = freq / elapsed;
			std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;
		}

		// show results
		for (std::vector<cv::Mat>::const_iterator cit = output_images.begin(); cit != output_images.end(); ++cit)
		{
			cv::imshow("undistorted RGB image", *cit);

			const unsigned char key = cv::waitKey(0);
			if (27 == key)
				break;
		}
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
