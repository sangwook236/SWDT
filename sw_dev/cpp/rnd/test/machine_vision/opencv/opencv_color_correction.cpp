//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


namespace {
namespace local {

void calc_color_correction_matrix_1(const cv::Mat &trueColorMat, const cv::Mat &actualColorMat, cv::Mat &correctionMatrix)
{
	const int IMG_WIDTH = trueColorMat.cols, IMG_HEIGHT = trueColorMat.rows;

	const cv::Mat I_true(trueColorMat.reshape(1, IMG_HEIGHT * IMG_WIDTH));
	const cv::Mat I_actual(actualColorMat.reshape(1, IMG_HEIGHT * IMG_WIDTH));
	cv::Mat I_actual_aug(cv::Mat::ones(IMG_HEIGHT * IMG_WIDTH, 4, CV_32FC1));

	// 3 x 3 matrix
	correctionMatrix = (I_actual.inv(cv::DECOMP_SVD) * I_true).t();

#if 0
	std::cout << "M = " << correctionMatrix << std::endl;

	std::cout << cv::norm(I_true.t(), correctionMatrix * I_actual.t(), cv::NORM_L2) << std::endl;

	std::cout << (I_true.t())(cv::Range::all(), cv::Range(0,10)) << std::endl;
	std::cout << (I_actual.t())(cv::Range::all(), cv::Range(0,10)) << std::endl;
	std::cout << (correctionMatrix * I_actual.t())(cv::Range::all(), cv::Range(0,10)) << std::endl;

	std::cout << (I_true.t() - correctionMatrix * I_actual.t())(cv::Range::all(), cv::Range(0,10)) << std::endl;
#endif
}

void calc_color_correction_matrix_2(const cv::Mat &trueColorMat, const cv::Mat &actualColorMat, cv::Mat &correctionMatrix)
{
	const int IMG_WIDTH = trueColorMat.cols, IMG_HEIGHT = trueColorMat.rows;

	const cv::Mat I_true(trueColorMat.reshape(1, IMG_HEIGHT * IMG_WIDTH));
	const cv::Mat I_actual(actualColorMat.reshape(1, IMG_HEIGHT * IMG_WIDTH));
	cv::Mat I_actual_aug(cv::Mat::ones(IMG_HEIGHT * IMG_WIDTH, 4, CV_32FC1));
	//I_actual.assignTo(I_actual_aug(cv::Range::all(), cv::Range(0, 3)));  // error !!! not working
#if defined(__GNUC__)
    cv::Mat I_actual_aug2(I_actual_aug, cv::Range::all(), cv::Range(0, 3));
	I_actual.copyTo(I_actual_aug);  // apply when the same type
#else
	I_actual.copyTo(I_actual_aug(cv::Range::all(), cv::Range(0, 3)));  // apply when the same type
#endif

	// 3 x 4 matrix
	correctionMatrix = (I_actual_aug.inv(cv::DECOMP_SVD) * I_true).t();

#if 0
	std::cout << "M_aug = " << correctionMatrix << std::endl;

	std::cout << cv::norm(I_true.t(), correctionMatrix * I_actual_aug.t(), cv::NORM_L2) << std::endl;

	std::cout << (I_true.t())(cv::Range::all(), cv::Range(0, 10)) << std::endl;
	std::cout << (I_actual.t())(cv::Range::all(), cv::Range(0,1 0)) << std::endl;
	std::cout << (correctionMatrix * I_actual_aug.t())(cv::Range::all(), cv::Range(0,10)) << std::endl;

	std::cout << (I_true.t() - correctionMatrix * I_actual_aug.t())(cv::Range::all(), cv::Range(0, 10)) << std::endl;
#endif
}

void color_correction()
{
	const int IMG_WIDTH = 100, IMG_HEIGHT = 100;
	const double noiseStdDev = 5.0;

	cv::Mat trueColorMat(IMG_HEIGHT, IMG_WIDTH, CV_32FC3);
	cv::Mat actualColorMat(IMG_HEIGHT, IMG_WIDTH, CV_32FC3);
	cv::Mat noiseColorMat(IMG_HEIGHT, IMG_WIDTH, CV_32FC3);

	cv::randu(trueColorMat, cv::Scalar::all(0), cv::Scalar::all(256));
    cv::randn(noiseColorMat, cv::Scalar::all(0), cv::Scalar::all(noiseStdDev));
	actualColorMat = trueColorMat + noiseColorMat;

	//
	cv::Mat M;
	//calc_color_correction_matrix_1(trueColorMat, actualColorMat, M);
	calc_color_correction_matrix_2(trueColorMat, actualColorMat, M);

	std::cout << "M = " << M << std::endl;
}

void color_correction_test_1()
{
#if 0
	const std::string filename("machine_vision_data\\opencv\\color_correction_0.png");
	const int PATCH_WIDTH = 200, PATCH_HEIGHT = 200;
	const int WHITE_BALANCE_PATCH_NUM = 3;
	const int COLOR_CORRECTION_PATCH_NUM = 3;
	const int PATCH_NUM = WHITE_BALANCE_PATCH_NUM + COLOR_CORRECTION_PATCH_NUM;
	const cv::Rect blackRect(50, 50, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect grayRect(360, 50, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect whiteRect(670, 50, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect redRect(50, 360, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect greenRect(360, 360, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect blueRect(670, 360, PATCH_WIDTH, PATCH_HEIGHT);
#elif 1
	const std::string filename("machine_vision_data\\opencv\\color_correction_1.jpg");
	const int PATCH_WIDTH = 450, PATCH_HEIGHT = 450;
	const int WHITE_BALANCE_PATCH_NUM = 3;
	const int COLOR_CORRECTION_PATCH_NUM = 3;
	const int PATCH_NUM = WHITE_BALANCE_PATCH_NUM + COLOR_CORRECTION_PATCH_NUM;
	const cv::Rect blackRect(240, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect grayRect(1050, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect whiteRect(1870, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect redRect(240, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect greenRect(1050, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect blueRect(1870, 1210, PATCH_WIDTH, PATCH_HEIGHT);
#elif 0
	const std::string filename("machine_vision_data\\opencv\\color_correction_2.jpg");
	const int PATCH_WIDTH = 450, PATCH_HEIGHT = 450;
	const int WHITE_BALANCE_PATCH_NUM = 3;
	const int COLOR_CORRECTION_PATCH_NUM = 3;
	const int PATCH_NUM = WHITE_BALANCE_PATCH_NUM + COLOR_CORRECTION_PATCH_NUM;
	const cv::Rect blackRect(240, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect grayRect(1050, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect whiteRect(1870, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect redRect(240, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect greenRect(1050, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect blueRect(1870, 1210, PATCH_WIDTH, PATCH_HEIGHT);
#elif 0
	const std::string filename("machine_vision_data\\opencv\\color_correction_3.jpg");
	const int PATCH_WIDTH = 450, PATCH_HEIGHT = 450;
	const int WHITE_BALANCE_PATCH_NUM = 3;
	const int COLOR_CORRECTION_PATCH_NUM = 3;
	const int PATCH_NUM = WHITE_BALANCE_PATCH_NUM + COLOR_CORRECTION_PATCH_NUM;
	const cv::Rect blackRect(240, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect grayRect(1050, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect whiteRect(1870, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect redRect(240, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect greenRect(1050, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect blueRect(1870, 1210, PATCH_WIDTH, PATCH_HEIGHT);
#endif

	const cv::Mat &image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);

	cv::Mat ref_white_balance_patch_images(PATCH_HEIGHT, PATCH_WIDTH * WHITE_BALANCE_PATCH_NUM, CV_32FC3);
	ref_white_balance_patch_images(cv::Range::all(), cv::Range(0,PATCH_WIDTH)).setTo(cv::Scalar::all(0));
	ref_white_balance_patch_images(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)).setTo(cv::Scalar::all(128));
	ref_white_balance_patch_images(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)).setTo(cv::Scalar::all(255));
	cv::Mat ref_color_correction_patch_mat(PATCH_HEIGHT, PATCH_WIDTH * COLOR_CORRECTION_PATCH_NUM, CV_32FC3);
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(0,PATCH_WIDTH)).setTo(cv::Scalar(0,0,255));
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)).setTo(cv::Scalar(0,255,0));
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)).setTo(cv::Scalar(255,0,0));

#if 0
	// error !!! cannot apply cv::Mat::copyTo() when different type
	cv::Mat actual_white_balance_patch_images(PATCH_HEIGHT, PATCH_WIDTH * WHITE_BALANCE_PATCH_NUM, CV_32FC3);
	image(blackRect).copyTo(actual_white_balance_patch_images(cv::Range::all(), cv::Range(0,PATCH_WIDTH)));
	image(grayRect).copyTo(actual_white_balance_patch_images(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)));
	image(whiteRect).copyTo(actual_white_balance_patch_images(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)));
	cv::Mat actual_color_correction_patch_mat(PATCH_HEIGHT, PATCH_WIDTH * COLOR_CORRECTION_PATCH_NUM, CV_32FC3);
	image(redRect).copyTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(0,PATCH_WIDTH)));
	image(greenRect).copyTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)));
	image(blueRect).copyTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)));
#else
#if defined(__GNUC__)
	cv::Mat actual_white_balance_patch_images(PATCH_HEIGHT, PATCH_WIDTH * WHITE_BALANCE_PATCH_NUM, CV_32FC3);
	{
        cv::Mat awbpi(actual_white_balance_patch_images, cv::Range::all(), cv::Range(0,PATCH_WIDTH));
        image(blackRect).convertTo(awbpi, CV_32FC3, 1, 0);
	}
	{
        cv::Mat awbpi(actual_white_balance_patch_images, cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH));
        image(grayRect).convertTo(awbpi, CV_32FC3, 1, 0);
	}
	{
        cv::Mat awbpi(actual_white_balance_patch_images, cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH));
        image(whiteRect).convertTo(awbpi, CV_32FC3, 1, 0);
	}
	cv::Mat actual_color_correction_patch_mat(PATCH_HEIGHT, PATCH_WIDTH * COLOR_CORRECTION_PATCH_NUM, CV_32FC3);
	{
        cv::Mat accpm(actual_color_correction_patch_mat, cv::Range::all(), cv::Range(0,PATCH_WIDTH));
        image(redRect).convertTo(accpm, CV_32FC3, 1, 0);
	}
	{
        cv::Mat accpm(actual_color_correction_patch_mat, cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH));
        image(greenRect).convertTo(accpm, CV_32FC3, 1, 0);
	}
	{
        cv::Mat accpm(actual_color_correction_patch_mat, cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH));
        image(blueRect).convertTo(accpm, CV_32FC3, 1, 0);
	}
#else
	cv::Mat actual_white_balance_patch_images(PATCH_HEIGHT, PATCH_WIDTH * WHITE_BALANCE_PATCH_NUM, CV_32FC3);
	image(blackRect).convertTo(actual_white_balance_patch_images(cv::Range::all(), cv::Range(0,PATCH_WIDTH)), CV_32FC3, 1, 0);
	image(grayRect).convertTo(actual_white_balance_patch_images(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)), CV_32FC3, 1, 0);
	image(whiteRect).convertTo(actual_white_balance_patch_images(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)), CV_32FC3, 1, 0);
	cv::Mat actual_color_correction_patch_mat(PATCH_HEIGHT, PATCH_WIDTH * COLOR_CORRECTION_PATCH_NUM, CV_32FC3);
	image(redRect).convertTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(0,PATCH_WIDTH)), CV_32FC3, 1, 0);
	image(greenRect).convertTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)), CV_32FC3, 1, 0);
	image(blueRect).convertTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)), CV_32FC3, 1, 0);
#endif
#endif

#if 0
	cv::Mat tmp;
	ref_white_balance_patch_images(cv::Range::all(), cv::Range(0,PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	ref_white_balance_patch_images(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	ref_white_balance_patch_images(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(0,PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);

	actual_white_balance_patch_images(cv::Range::all(), cv::Range(0,PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	actual_white_balance_patch_images(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	actual_white_balance_patch_images(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	actual_color_correction_patch_mat(cv::Range::all(), cv::Range(0,PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	actual_color_correction_patch_mat(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
	actual_color_correction_patch_mat(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)).convertTo(tmp, CV_8UC3, 1, 0);
	cv::imshow(windowName2, tmp);  cv::waitKey(0);
#endif

	//
	cv::Mat actual_patched_images(2 * PATCH_HEIGHT, std::max(WHITE_BALANCE_PATCH_NUM, COLOR_CORRECTION_PATCH_NUM) * PATCH_WIDTH, CV_8UC3);
#if defined(__GNUC__)
	{
	    cv::Mat api(actual_patched_images, cv::Range(0,PATCH_HEIGHT), cv::Range(0,PATCH_WIDTH));
        image(blackRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(0,PATCH_HEIGHT), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH));
        image(grayRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(0,PATCH_HEIGHT), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH));
        image(whiteRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(0,PATCH_WIDTH));
        image(redRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH));
        image(greenRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH));
        image(blueRect).convertTo(api, CV_8UC3, 1, 0);
	}
#else
	image(blackRect).convertTo(actual_patched_images(cv::Range(0,PATCH_HEIGHT), cv::Range(0,PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(grayRect).convertTo(actual_patched_images(cv::Range(0,PATCH_HEIGHT), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(whiteRect).convertTo(actual_patched_images(cv::Range(0,PATCH_HEIGHT), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(redRect).convertTo(actual_patched_images(cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(0,PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(greenRect).convertTo(actual_patched_images(cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(blueRect).convertTo(actual_patched_images(cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)), CV_8UC3, 1, 0);
#endif

	cv::Mat corrected_patch_images(2 * PATCH_HEIGHT, std::max(WHITE_BALANCE_PATCH_NUM, COLOR_CORRECTION_PATCH_NUM) * PATCH_WIDTH, CV_8UC3);
	cv::Mat corrected_image;
	const double t = (double)cv::getTickCount();
	{
		cv::Mat M_wb, M_cc;

		// calculate white balance matrix
		calc_color_correction_matrix_1(ref_white_balance_patch_images, actual_white_balance_patch_images, M_wb);
#if 1
		// calculate color correction matrix
		calc_color_correction_matrix_1(ref_color_correction_patch_mat, actual_color_correction_patch_mat, M_cc);
		//const cv::Mat M_wc(M_wb * M_cc);  // incorrectly working
		const cv::Mat M_wc(M_cc);

		// apply the white balance & color correction matrices to input patch images
		const cv::Mat corrected_patch_mat1(cv::Mat(actual_white_balance_patch_images.reshape(1, actual_white_balance_patch_images.rows * actual_white_balance_patch_images.cols) * M_wc.t()).reshape(3, PATCH_HEIGHT));
		const cv::Mat corrected_patch_mat2(cv::Mat(actual_color_correction_patch_mat.reshape(1, actual_color_correction_patch_mat.rows * actual_color_correction_patch_mat.cols) * M_wc.t()).reshape(3, PATCH_HEIGHT));

		// apply the color correction matrix to input image
		cv::Mat half_image;
		cv::resize(image, half_image, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
		cv::Mat image_mat(half_image.size(), CV_32FC3);
		half_image.convertTo(image_mat, CV_32FC3, 1, 0);
		cv::Mat(image_mat.reshape(1, image_mat.rows * image_mat.cols) * M_wc.t()).reshape(3, image_mat.rows).convertTo(corrected_image, CV_8UC3, 1, 0);
#else
		// calculate color correction matrix
		calc_color_correction_matrix_2(ref_color_correction_patch_mat, actual_color_correction_patch_mat, M_cc);
		//const cv::Mat M_wc(M_wb * M_cc);  // incorrectly working
		const cv::Mat M_wc(M_cc);  // incorrectly working

		// apply the white balance & color correction matrices to input patch images
		cv::Mat actual_white_balance_mat_aug(cv::Mat::ones(actual_white_balance_patch_images.rows * actual_white_balance_patch_images.cols, 4, CV_32FC1));
		actual_white_balance_patch_images.reshape(1, actual_white_balance_patch_images.rows * actual_white_balance_patch_images.cols).copyTo(actual_white_balance_mat_aug(cv::Range::all(), cv::Range(0,3)));
		const cv::Mat corrected_patch_mat1(cv::Mat(actual_white_balance_mat_aug * M_wc.t()).reshape(3, PATCH_HEIGHT));

		cv::Mat actual_color_correction_mat_aug(cv::Mat::ones(actual_color_correction_patch_mat.rows * actual_color_correction_patch_mat.cols, 4, CV_32FC1));
		actual_color_correction_patch_mat.reshape(1, actual_color_correction_patch_mat.rows * actual_color_correction_patch_mat.cols).copyTo(actual_color_correction_mat_aug(cv::Range::all(), cv::Range(0,3)));
		const cv::Mat corrected_patch_mat2(cv::Mat(actual_color_correction_mat_aug * M_wc.t()).reshape(3, PATCH_HEIGHT));

		// apply the color correction matrix to input image
		cv::Mat half_image;
		cv::resize(image, half_image, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
		cv::Mat image_mat_aug(cv::Mat::ones(half_image.rows * half_image.cols, 4, CV_32FC1));
		half_image.reshape(1, half_image.rows * half_image.cols).convertTo(image_mat_aug(cv::Range::all(), cv::Range(0,3)), CV_32FC3, 1, 0);
		cv::Mat(image_mat_aug * M_wc.t()).reshape(3, half_image.rows).convertTo(corrected_image, CV_8UC3, 1, 0);
#endif
		std::cout << "white balance matrix = " << M_wb << std::endl;
		std::cout << "color correction matrix = " << M_cc << std::endl;
		std::cout << "total correction matrix = " << M_wc << std::endl;

#if defined(__GNUC__)
        cv::Mat cpm_tmp1(corrected_patch_images, cv::Range(0,PATCH_HEIGHT), cv::Range::all());
		corrected_patch_mat1.convertTo(cpm_tmp1, CV_8UC3, 1, 0);
		cv::Mat cpm_tmp2(corrected_patch_images, cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range::all());
		corrected_patch_mat2.convertTo(cpm_tmp2, CV_8UC3, 1, 0);
#else
		corrected_patch_mat1.convertTo(corrected_patch_images(cv::Range(0,PATCH_HEIGHT), cv::Range::all()), CV_8UC3, 1, 0);
		corrected_patch_mat2.convertTo(corrected_patch_images(cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range::all()), CV_8UC3, 1, 0);
#endif
	}
	const double et = ((double)cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();
	std::cout << "time elapsed: " << et << "ms" << std::endl;

	//
	const std::string windowName1("color correction - input image");
	const std::string windowName2("color correction - corrected patch image");
	const std::string windowName3("color correction - corrected image");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName3, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName1, actual_patched_images);
	cv::imshow(windowName2, corrected_patch_images);
	cv::imshow(windowName3, corrected_image);

	cv::waitKey(0);

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
	cv::destroyWindow(windowName3);
}

void color_correction_test_2()
{
#if 0
	const std::string filename("machine_vision_data\\opencv\\color_correction_0.png");
	const int PATCH_WIDTH = 200, PATCH_HEIGHT = 200;
	const int WHITE_BALANCE_PATCH_NUM = 3;
	const int COLOR_CORRECTION_PATCH_NUM = 3;
	const int PATCH_NUM = WHITE_BALANCE_PATCH_NUM + COLOR_CORRECTION_PATCH_NUM;
	const cv::Rect blackRect(50, 50, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect grayRect(360, 50, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect whiteRect(670, 50, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect redRect(50, 360, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect greenRect(360, 360, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect blueRect(670, 360, PATCH_WIDTH, PATCH_HEIGHT);
#elif 1
	const std::string filename("machine_vision_data\\opencv\\color_correction_1.jpg");
	const int PATCH_WIDTH = 450, PATCH_HEIGHT = 450;
	const int WHITE_BALANCE_PATCH_NUM = 3;
	const int COLOR_CORRECTION_PATCH_NUM = 3;
	const int PATCH_NUM = WHITE_BALANCE_PATCH_NUM + COLOR_CORRECTION_PATCH_NUM;
	const cv::Rect blackRect(240, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect grayRect(1050, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect whiteRect(1870, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect redRect(240, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect greenRect(1050, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect blueRect(1870, 1210, PATCH_WIDTH, PATCH_HEIGHT);
#elif 0
	const std::string filename("machine_vision_data\\opencv\\color_correction_2.jpg");
	const int PATCH_WIDTH = 450, PATCH_HEIGHT = 450;
	const int WHITE_BALANCE_PATCH_NUM = 3;
	const int COLOR_CORRECTION_PATCH_NUM = 3;
	const int PATCH_NUM = WHITE_BALANCE_PATCH_NUM + COLOR_CORRECTION_PATCH_NUM;
	const cv::Rect blackRect(240, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect grayRect(1050, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect whiteRect(1870, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect redRect(240, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect greenRect(1050, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect blueRect(1870, 1210, PATCH_WIDTH, PATCH_HEIGHT);
#elif 0
	const std::string filename("machine_vision_data\\opencv\\color_correction_3.jpg");
	const int PATCH_WIDTH = 450, PATCH_HEIGHT = 450;
	const int WHITE_BALANCE_PATCH_NUM = 3;
	const int COLOR_CORRECTION_PATCH_NUM = 3;
	const int PATCH_NUM = WHITE_BALANCE_PATCH_NUM + COLOR_CORRECTION_PATCH_NUM;
	const cv::Rect blackRect(240, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect grayRect(1050, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect whiteRect(1870, 390, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect redRect(240, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect greenRect(1050, 1210, PATCH_WIDTH, PATCH_HEIGHT);
	const cv::Rect blueRect(1870, 1210, PATCH_WIDTH, PATCH_HEIGHT);
#endif

	const cv::Mat &image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);

	cv::Mat ref_color_correction_patch_mat(PATCH_HEIGHT, PATCH_WIDTH * PATCH_NUM, CV_32FC3);
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(0,PATCH_WIDTH)).setTo(cv::Scalar::all(0));
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)).setTo(cv::Scalar::all(128));
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)).setTo(cv::Scalar::all(255));
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(3*PATCH_WIDTH,4*PATCH_WIDTH)).setTo(cv::Scalar(0,0,255));
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(4*PATCH_WIDTH,5*PATCH_WIDTH)).setTo(cv::Scalar(0,255,0));
	ref_color_correction_patch_mat(cv::Range::all(), cv::Range(5*PATCH_WIDTH,6*PATCH_WIDTH)).setTo(cv::Scalar(255,0,0));

#if 0
	// error !!! cannot apply cv::Mat::copyTo() when different type
	cv::Mat actual_color_correction_patch_mat(PATCH_HEIGHT, PATCH_WIDTH * PATCH_NUM, CV_32FC3);
	image(blackRect).copyTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(0,PATCH_WIDTH)));
	image(grayRect).copyTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)));
	image(whiteRect).copyTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)));
	image(redRect).copyTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(3*PATCH_WIDTH,4*PATCH_WIDTH)));
	image(greenRect).copyTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(4*PATCH_WIDTH,5*PATCH_WIDTH)));
	image(blueRect).copyTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(5*PATCH_WIDTH,6*PATCH_WIDTH)));
#else
#if defined(__GNUC__)
	cv::Mat actual_color_correction_patch_mat(PATCH_HEIGHT, PATCH_WIDTH * PATCH_NUM, CV_32FC3);
	{
        cv::Mat accpm(actual_color_correction_patch_mat, cv::Range::all(), cv::Range(0,PATCH_WIDTH));
        image(blackRect).convertTo(accpm, CV_32FC3, 1, 0);
	}
	{
        cv::Mat accpm(actual_color_correction_patch_mat, cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH));
        image(grayRect).convertTo(accpm, CV_32FC3, 1, 0);
	}
	{
        cv::Mat accpm(actual_color_correction_patch_mat, cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH));
        image(whiteRect).convertTo(accpm, CV_32FC3, 1, 0);
	}
	{
        cv::Mat accpm(actual_color_correction_patch_mat, cv::Range::all(), cv::Range(3*PATCH_WIDTH,4*PATCH_WIDTH));
        image(redRect).convertTo(accpm, CV_32FC3, 1, 0);
	}
	{
        cv::Mat accpm(actual_color_correction_patch_mat, cv::Range::all(), cv::Range(4*PATCH_WIDTH,5*PATCH_WIDTH));
        image(greenRect).convertTo(accpm, CV_32FC3, 1, 0);
	}
	{
        cv::Mat accpm(actual_color_correction_patch_mat, cv::Range::all(), cv::Range(5*PATCH_WIDTH,6*PATCH_WIDTH));
        image(blueRect).convertTo(accpm, CV_32FC3, 1, 0);
	}
#else
	cv::Mat actual_color_correction_patch_mat(PATCH_HEIGHT, PATCH_WIDTH * PATCH_NUM, CV_32FC3);
	image(blackRect).convertTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(0,PATCH_WIDTH)), CV_32FC3, 1, 0);
	image(grayRect).convertTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)), CV_32FC3, 1, 0);
	image(whiteRect).convertTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)), CV_32FC3, 1, 0);
	image(redRect).convertTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(3*PATCH_WIDTH,4*PATCH_WIDTH)), CV_32FC3, 1, 0);
	image(greenRect).convertTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(4*PATCH_WIDTH,5*PATCH_WIDTH)), CV_32FC3, 1, 0);
	image(blueRect).convertTo(actual_color_correction_patch_mat(cv::Range::all(), cv::Range(5*PATCH_WIDTH,6*PATCH_WIDTH)), CV_32FC3, 1, 0);
#endif
#endif

	//
	cv::Mat actual_patched_images(2 * PATCH_HEIGHT, std::max(WHITE_BALANCE_PATCH_NUM, COLOR_CORRECTION_PATCH_NUM) * PATCH_WIDTH, CV_8UC3);
#if defined(__GNUC__)
	{
	    cv::Mat api(actual_patched_images, cv::Range(0,PATCH_HEIGHT), cv::Range(0,PATCH_WIDTH));
        image(blackRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(0,PATCH_HEIGHT), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH));
        image(grayRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(0,PATCH_HEIGHT), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH));
        image(whiteRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(0,PATCH_WIDTH));
        image(redRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH));
        image(greenRect).convertTo(api, CV_8UC3, 1, 0);
	}
	{
	    cv::Mat api(actual_patched_images, cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH));
        image(blueRect).convertTo(api, CV_8UC3, 1, 0);
	}
#else
	image(blackRect).convertTo(actual_patched_images(cv::Range(0,PATCH_HEIGHT), cv::Range(0,PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(grayRect).convertTo(actual_patched_images(cv::Range(0,PATCH_HEIGHT), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(whiteRect).convertTo(actual_patched_images(cv::Range(0,PATCH_HEIGHT), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(redRect).convertTo(actual_patched_images(cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(0,PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(greenRect).convertTo(actual_patched_images(cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(PATCH_WIDTH,2*PATCH_WIDTH)), CV_8UC3, 1, 0);
	image(blueRect).convertTo(actual_patched_images(cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range(2*PATCH_WIDTH,3*PATCH_WIDTH)), CV_8UC3, 1, 0);
#endif

	cv::Mat corrected_patch_images(2 * PATCH_HEIGHT, std::max(WHITE_BALANCE_PATCH_NUM, COLOR_CORRECTION_PATCH_NUM) * PATCH_WIDTH, CV_8UC3);
	cv::Mat corrected_image;
	const double t = (double)cv::getTickCount();
	{
		cv::Mat M;

#if 1
		// calculate color correction matrix
		calc_color_correction_matrix_1(ref_color_correction_patch_mat, actual_color_correction_patch_mat, M);

		// apply the color correction matrix to input patch images
		const cv::Mat corrected_patch_mat(cv::Mat(actual_color_correction_patch_mat.reshape(1, actual_color_correction_patch_mat.rows * actual_color_correction_patch_mat.cols) * M.t()).reshape(3, PATCH_HEIGHT));

		// apply the color correction matrix to input image
		cv::Mat half_image;
		cv::resize(image, half_image, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
		cv::Mat image_mat(half_image.size(), CV_32FC3);
		half_image.convertTo(image_mat, CV_32FC3, 1, 0);
		cv::Mat(image_mat.reshape(1, image_mat.rows * image_mat.cols) * M.t()).reshape(3, image_mat.rows).convertTo(corrected_image, CV_8UC3, 1, 0);
#else
		// calculate color correction matrix
		calc_color_correction_matrix_2(ref_color_correction_patch_mat, actual_color_correction_patch_mat, M);  // incorrectly working

		// apply the color correction matrix to input patch images
		cv::Mat actual_color_correction_mat_aug(cv::Mat::ones(actual_color_correction_patch_mat.rows * actual_color_correction_patch_mat.cols, 4, CV_32FC1));
		actual_color_correction_patch_mat.reshape(1, actual_color_correction_patch_mat.rows * actual_color_correction_patch_mat.cols).copyTo(actual_color_correction_mat_aug(cv::Range::all(), cv::Range(0,3)));
		const cv::Mat corrected_patch_mat(cv::Mat(actual_color_correction_mat_aug * M.t()).reshape(3, PATCH_HEIGHT));

		// apply the color correction matrix to input image
		cv::Mat half_image;
		cv::resize(image, half_image, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
		cv::Mat image_mat_aug(cv::Mat::ones(half_image.rows * half_image.cols, 4, CV_32FC1));
		half_image.reshape(1, half_image.rows * half_image.cols).convertTo(image_mat_aug(cv::Range::all(), cv::Range(0,3)), CV_32FC3, 1, 0);
		cv::Mat(image_mat_aug * M.t()).reshape(3, half_image.rows).convertTo(corrected_image, CV_8UC3, 1, 0);
#endif
		std::cout << "color correction matrix = " << M << std::endl;

#if defined(__GNUC__)
        cv::Mat cpm_tmp1(corrected_patch_images, cv::Range(0,PATCH_HEIGHT), cv::Range::all());
		corrected_patch_mat(cv::Range::all(), cv::Range(0,3*PATCH_HEIGHT)).convertTo(cpm_tmp1, CV_8UC3, 1, 0);
        cv::Mat cpm_tmp2(corrected_patch_images, cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range::all());
		corrected_patch_mat(cv::Range::all(), cv::Range(3*PATCH_HEIGHT,6*PATCH_HEIGHT)).convertTo(cpm_tmp2, CV_8UC3, 1, 0);
#else
		corrected_patch_mat(cv::Range::all(), cv::Range(0,3*PATCH_HEIGHT)).convertTo(corrected_patch_images(cv::Range(0,PATCH_HEIGHT), cv::Range::all()), CV_8UC3, 1, 0);
		//corrected_patch_mat(cv::Range::all(), cv::Range(3*PATCH_HEIGHT,6*PATCH_HEIGHT)).convertTo(corrected_patch_images(cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range::all()), CV_8UC3, 1, 0);
		corrected_patch_mat(cv::Range::all(), cv::Range(3*PATCH_HEIGHT,6*PATCH_HEIGHT)).convertTo(corrected_patch_images(cv::Range(PATCH_HEIGHT,2*PATCH_HEIGHT), cv::Range::all()), CV_8UC3, 1, 0);
#endif
	}
	const double et = ((double)cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();
	std::cout << "time elapsed: " << et << "ms" << std::endl;

	//
	const std::string windowName1("color correction - input image");
	const std::string windowName2("color correction - corrected patch image");
	const std::string windowName3("color correction - corrected image");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName3, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName1, actual_patched_images);
	cv::imshow(windowName2, corrected_patch_images);
	cv::imshow(windowName3, corrected_image);

	cv::waitKey(0);

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
	cv::destroyWindow(windowName3);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void color_correction()
{
	//local::color_correction();
	//local::color_correction_test_1();
	local::color_correction_test_2();
}

}  // namespace my_opencv
