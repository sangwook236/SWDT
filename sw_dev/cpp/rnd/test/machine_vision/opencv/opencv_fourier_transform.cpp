//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

void convolute_by_dft(const cv::Mat &A, const cv::Mat &B, cv::Mat &C)
{
	// reallocate the output array if needed
	C.create(std::abs(A.rows - B.rows) + 1, std::abs(A.cols - B.cols) + 1, A.type());

	// compute the size of DFT transform
	cv::Size dftSize;
	dftSize.width = cv::getOptimalDFTSize(A.cols + B.cols - 1);
	dftSize.height = cv::getOptimalDFTSize(A.rows + B.rows - 1);

	// allocate temporary buffers and initialize them with 0's
	cv::Mat tempA(dftSize, A.type(), cv::Scalar::all(0));
	cv::Mat tempB(dftSize, B.type(), cv::Scalar::all(0));

	// copy A and B to the top-left corners of tempA and tempB, respectively
	cv::Mat roiA(tempA, cv::Rect(0, 0, A.cols, A.rows));
	A.copyTo(roiA);
	cv::Mat roiB(tempB, cv::Rect(0, 0, B.cols, B.rows));
	B.copyTo(roiB);

	// now transform the padded A & B in-place;
	// use "nonzeroRows" hint for faster processing
	cv::dft(tempA, tempA, 0, A.rows);
	cv::dft(tempB, tempB, 0, B.rows);

	// multiply the spectrums;
	// the function handles packed spectr
	// transform the product back from the frequency domain.
	// Even though all the result rows will be non-zero,
	// we need only the first C.rows of them, and thus we pass nonzeroRows == C.rows
	cv::dft(tempA, tempA, cv::DFT_INVERSE + cv::DFT_SCALE, C.rows);

	// now copy the result back to C.
	tempA(cv::Rect(0, 0, C.cols, C.rows)).copyTo(C);

	// all the temporary buffers will be deallocated automatically
}

void dft_based_convolution()
{
	std::list<std::string> filenames;
	filenames.push_back("machine_vision_data\\opencv\\pic1.png");
	filenames.push_back("machine_vision_data\\opencv\\pic2.png");
	filenames.push_back("machine_vision_data\\opencv\\pic3.png");
	filenames.push_back("machine_vision_data\\opencv\\pic4.png");
	filenames.push_back("machine_vision_data\\opencv\\pic5.png");
	filenames.push_back("machine_vision_data\\opencv\\pic6.png");
	filenames.push_back("machine_vision_data\\opencv\\stuff.jpg");
	filenames.push_back("machine_vision_data\\opencv\\synthetic_face.png");
	filenames.push_back("machine_vision_data\\opencv\\puzzle.png");
	filenames.push_back("machine_vision_data\\opencv\\fruits.jpg");
	filenames.push_back("machine_vision_data\\opencv\\lena_rgb.bmp");
	filenames.push_back("machine_vision_data\\opencv\\hand_01.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_05.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_24.jpg");

	const std::string windowName1("Fourier transform - #1");
	const std::string windowName2("Fourier transform - #2");
	const std::string windowName3("Fourier transform - #3");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName3, cv::WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			//cv::cvtColor(img, gray, CV_RGB2GRAY);

		cv::Mat img_float(gray.size(), CV_64FC1);
		gray.convertTo(img_float, CV_64FC1, 1.0 / 255.0, 0.0);

#if 0
		const int kernel_size = 11;
		const cv::Mat &kernel = cv::getGaussianKernel(kernel_size, -1.0, CV_64F);
#else
		const int kernel_size = 3;
		cv::Mat kernel(kernel_size, kernel_size, CV_64FC1);
		kernel.at<double>(0, 0) = -1.0;  kernel.at<double>(0, 1) = 0.0;  kernel.at<double>(0, 2) = 1.0;
		kernel.at<double>(1, 0) = -2.0;  kernel.at<double>(1, 1) = 0.0;  kernel.at<double>(1, 2) = 2.0;
		kernel.at<double>(2, 0) = -1.0;  kernel.at<double>(2, 1) = 0.0;  kernel.at<double>(2, 2) = 1.0;
#endif

		// FIXME [modify] >>
		// caution: not correctly working

		cv::Mat kernel2;
		cv::flip(kernel, kernel2, -1);

		cv::Mat result1;
		cv::filter2D(img_float, result1, -1, kernel2, cv::Point(kernel2.cols - kernel2.cols/2 - 1, kernel2.rows - kernel2.rows/2 - 1), 0, cv::BORDER_DEFAULT);

		cv::Mat result2;
		convolute_by_dft(img_float, kernel, result2);

		double minVal, maxVal;
		cv::minMaxLoc(result1, &minVal, &maxVal);
		cv::Mat result_gray1(result1.size(), CV_8UC1);
		const double a1 = 255.0 / (maxVal - minVal), b1 = -a1 * minVal;
		result1.convertTo(result_gray1, CV_8UC1, a1, b1);
		cv::minMaxLoc(result2, &minVal, &maxVal);
		cv::Mat result_gray2(result2.size(), CV_8UC1);
		const double a2 = 255.0 / (maxVal - minVal), b2 = -a2 * minVal;
		result1.convertTo(result_gray2, CV_8UC1, a2, b2);

		//
		cv::imshow(windowName1, gray);
		cv::imshow(windowName2, result_gray1);
		cv::imshow(windowName3, result_gray2);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
	cv::destroyWindow(windowName3);
}

void sinusoidal_function()
{
	const double Fs = 1000.0;  // sampling frequency
	const double Ts = 1.0 / Fs;  // sampling time
	const size_t L = 1000;  // length of signal

	//cv::Mat time(1, L, CV_64FC1);
	cv::Mat x(1, L, CV_64FC1);

	for (size_t i = 0; i < L; ++i)
	{
		const double t = double(i) * Ts;
		//time.at<double>(0,i) = t;
		x.at<double>(0,i) = 0.7 * std::sin(2.0 * CV_PI * 50.0 * t) + std::sin(2.0 * CV_PI * 120.0 * t);  // 50 Hz + 120 Hz
	}

	// compute the size of DFT transform
#if 1
	// FIXME [check] >>
	//const cv::Size dftSize(cv::getOptimalDFTSize(x.cols - 1), 1);
	const cv::Size dftSize(cv::getOptimalDFTSize(x.cols), 1);
#else
	// 2^n >= L ==> n = log2(L)
	const int &nn = size_t(std::ceil(std::log(double(L)) / std::log(2.0)));
	const cv::Size dftSize(cvRound(std::pow(2.0, nn)), 1);
#endif

	// allocate temporary buffers and initialize them with 0's
	cv::Mat temp_x(dftSize, x.type(), cv::Scalar::all(0));

	cv::Mat x_roi(temp_x, cv::Rect(0, 0, x.cols, x.rows));
	x.copyTo(x_roi);

	cv::Mat X;
	//cv::dft(x, X, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT, x.rows);
	cv::dft(temp_x, X, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT, x.rows);

	std::vector<cv::Mat> real_imag;
	cv::split(X, real_imag);

	cv::Mat mag, phase;
	cv::magnitude(real_imag[0], real_imag[1], mag);
	//cv::phase(real_imag[0], real_imag[1], phase);

	// FIXME [check] >>
	// available frequency range: [0, Fs / 2] ==> 0.5 * Fs * [0, 1]
	const size_t L2 = dftSize.width / 2;

	const cv::Mat threshold_mag = mag > 0.1;
	for (size_t i = 0; i <= L2; ++i)
	{
		if (threshold_mag.at<unsigned char>(0,i))
		{
			const double &freq = double(i) / L2 * 0.5 * Fs;
			//const double &freq = double(i) / (L2 + 1) * 0.5 * Fs;
			std::cout << i << ": " << mag.at<double>(0,i) << ", " << freq << std::endl;
		}
	}
}

}  // namespace local
}  // unnamed namespace

namespace opencv {

void fourier_transform()
{
	//local::dft_based_convolution();

	local::sinusoidal_function();
}

}  // namespace opencv
