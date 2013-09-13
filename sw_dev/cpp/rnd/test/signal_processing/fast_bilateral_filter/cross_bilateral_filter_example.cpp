//#include "stdafx.h"
#define CHRONO
#include "../fast_bilateral_filter_lib/geom.h"
#include "../fast_bilateral_filter_lib/fast_lbf.h"
#include "../fast_bilateral_filter_lib/linear_bf.h"
//#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>


namespace {
namespace local {

void smooth_image(const cv::Mat &in, cv::Mat &out)
{
#if 0
	// METHOD #1: down-scale and up-scale the image to filter out the noise.

	{
		cv::Mat tmp;
		cv::pyrDown(in, tmp);
		cv::pyrUp(tmp, out);
	}
#elif 0
	// METHOD #2: Gaussian filtering.

	{
		// FIXME [adjust] >> adjust parameters.
		const int kernelSize = 3;
		const double sigma = 0;
		cv::GaussianBlur(in, out, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
	}
#elif 1
	// METHOD #3: box filtering.

	{
		// FIXME [adjust] >> adjust parameters.
		const int ddepth = -1;  // the output image depth. -1 to use src.depth().
		const int kernelSize = 5;
		const bool normalize = true;
		cv::boxFilter(in, out, ddepth, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_DEFAULT);
		//cv::blur(in, out, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), cv::BORDER_DEFAULT);  // use the normalized box filter.
	}
#elif 0
	// METHOD #4: bilateral filtering.

	{
		// FIXME [adjust] >> adjust parameters.
		const int diameter = -1;  // diameter of each pixel neighborhood that is used during filtering. if it is non-positive, it is computed from sigmaSpace.
		const double sigmaColor = 3.0;  // for range filter.
		const double sigmaSpace = 50.0;  // for space filter.
		cv::bilateralFilter(in, out, diameter, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
	}
#else
	// METHOD #5: no filtering.

	out = in;
#endif
}

void detect_edge(const cv::Mat &in, cv::Mat &out)
{
#if 0
	// METHOD #1: using Sobel operator.

	{
		// compute x- & y-gradients.
		cv::Mat xgradient, ygradient;

		//const int ksize = 5;
		const int ksize = CV_SCHARR;  // use Scharr operator
		cv::Sobel(in, xgradient, CV_32FC1, 1, 0, ksize, 1.0, 0.0, cv::BORDER_DEFAULT);
		cv::Sobel(in, ygradient, CV_32FC1, 0, 1, ksize, 1.0, 0.0, cv::BORDER_DEFAULT);

		cv::Mat gradient;
		cv::magnitude(xgradient, ygradient, gradient);

		double minVal, maxVal;
		cv::minMaxLoc(gradient, &minVal, &maxVal);

		const double truncation_ratio = 0.1;
#if 1
		gradient.setTo(cv::Scalar::all(0), gradient < truncation_ratio * maxVal);
		gradient.convertTo(out, CV_8UC1, 255.0 / maxVal, 0.0);
#else
		gradient.setTo(cv::Scalar::all(0), gradient < minVal + truncation_ratio * (maxVal - minVal));
		gradient.convertTo(out, CV_8UC1, 255.0 / (maxVal - minVal), -255.0 * minVal / (maxVal - minVal));
#endif
	}
#elif 0
	// METHOD #2: using derivative of Gaussian distribution.

	{
		cv::Mat img_double;
		double minVal, maxVal;
		cv::minMaxLoc(in, &minVal, &maxVal);
		in.convertTo(img_double, CV_64FC1, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));

		const double deriv_sigma = 3.0;
		const double sigma2 = deriv_sigma * deriv_sigma;
		const double _2sigma2 = 2.0 * sigma2;
		const double sigma3 = sigma2 * deriv_sigma;
		const double den = std::sqrt(2.0 * boost::math::constants::pi<double>()) * sigma3;

		const int deriv_kernel_size = 2 * (int)std::ceil(deriv_sigma) + 1;
		cv::Mat kernelX(1, deriv_kernel_size, CV_64FC1), kernelY(deriv_kernel_size, 1, CV_64FC1);

		// construct derivative kernels.
		for (int i = 0, k = -deriv_kernel_size/2; k <= deriv_kernel_size/2; ++i, ++k)
		{
			const double val = k * std::exp(-k*k / _2sigma2) / den;
			kernelX.at<double>(0, i) = val;
			kernelY.at<double>(i, 0) = val;
		}

		// compute x- & y-gradients.
		cv::Mat xgradient, ygradient;
		cv::filter2D(img_double, xgradient, -1, kernelX, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);
		cv::filter2D(img_double, ygradient, -1, kernelY, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);

		cv::Mat gradient;
		cv::magnitude(xgradient, ygradient, gradient);

		cv::minMaxLoc(gradient, &minVal, &maxVal);

		const double truncation_ratio = 0.0;
#if 1
		gradient.setTo(cv::Scalar::all(0), gradient < truncation_ratio * maxVal);
		gradient.convertTo(out, CV_8UC1, 255.0 / maxVal, 0.0);
#else
		gradient.setTo(cv::Scalar::all(0), gradient < minVal + truncation_ratio * (maxVal - minVal));
		gradient.convertTo(out, CV_8UC1, 255.0 / (maxVal - minVal), -255.0 * minVal / (maxVal - minVal));
#endif
	}
#elif 1
	// METHOD #3: using Canny edge detector.

	{
		const int lowerEdgeThreshold = 30, upperEdgeThreshold = 50;
		const int apertureSize = 3;  // aperture size for the Sobel() operator.
		const bool useL2 = true;  // if true, use L2 norm. otherwise, use L1 norm (faster).
		cv::Canny(in, out, lowerEdgeThreshold, upperEdgeThreshold, apertureSize, useL2);
	}
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_fast_bilateral_filter {

// [ref] ${FAST_BILATERAL_FILTER_HOME}/CROSS_BILATERAL_FILTER/cross_bilateral_filter.cpp
void cross_bilateral_filter_example()
{
	typedef Array_2D<double> image_type;

#if 0
	if (6 != argc)
	{
		std::cerr << "error: wrong arguments" << std::endl;
		std::cerr << std::endl;
		std::cerr << "usage: " << argv[0] << " input.ppm edge.ppm output.ppm sigma_s sigma_r" << std::endl;
		std::cerr << std::endl;
		std::cerr << "spatial parameter (measured in pixels)" << std::endl;
		std::cerr << "---------------------------------------" << std::endl;
		std::cerr << "sigma_s    : parameter of the bilateral filter (try 16)" << std::endl;
		std::cerr << std::endl;
		std::cerr << "range parameter (intensity is scaled to [0.0,1.0])" << std::endl;
		std::cerr << "---------------------------------------------------" << std::endl;
		std::cerr << "sigma_r    : parameter of the bilateral filter (try 0.1)" << std::endl;
		std::cerr << std::endl;
		return;
	}
#else
	const std::string input_filename("./data/signal_processing/fast_bilateral_filter/building.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/dome.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/dragon_hires.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/dragon_lores.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/flower.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/housecorner_hires.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/housecorner_lores.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/rock.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/swamp.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/synthetic.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/temple.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/turtle.ppm");
	
	const std::string output_filename("./data/signal_processing/fast_bilateral_filter/cross_bf_output.ppm");
	
	//const std::string high_freq_filename("./data/signal_processing/fast_bilateral_filter/high_freq_image.ppm");
	const std::string high_freq_filename;
	
	const double sigma_s = 5;  // space sigma.
	const double sigma_r = 0.1;  // range sigma.
#endif

	//---------------------------------------------------------------

	std::cout << "Load the input image '" << input_filename << "'... " << std::flush;

	std::ifstream ppm_in(input_filename.c_str(), std::ios::binary);

	std::string magic_number("  ");
	ppm_in.get(magic_number[0]);
	ppm_in.get(magic_number[1]);
	if (magic_number != std::string("P6"))
	{
		std::cerr << "error: unrecognized file format\n" << input_filename << " is not a PPM file.\n" << std::endl;
		return;
	}

	unsigned width, height, bpp;
	ppm_in >> width >> height >> bpp;
	if (255 != bpp)
	{
		std::cerr << "error: unsupported maximum value (" << bpp << ")\n" << "It must be 255." << std::endl;
		return;
	}

	image_type input_image(width, height);

	char ch;
	ppm_in.get(ch);  // Trailing white space.

	char r, g, b;
	for (unsigned y = 0; y < height; ++y)
	{
		for (unsigned x = 0; x < width; ++x)
		{
			ppm_in.get(r);
			ppm_in.get(g);
			ppm_in.get(b);

			const unsigned char R = static_cast<unsigned char>(r);
			const unsigned char G = static_cast<unsigned char>(g);
			const unsigned char B = static_cast<unsigned char>(b);

			input_image(x, y) = (20.0 * R + 40.0 * G + 1.0 * B) / (61.0 * 255.0); 
		}
	}

	ppm_in.close();

	std::cout << "Done" << std::endl;

	//---------------------------------------------------------------

	// image with high-frequency components like edges.
	image_type high_freq_image(width, height);
	cv::Mat high_freq_img;
#if 0
	// NOTICE [caution] >>
	//	An edge image is not sufficient to play a role in a high-frequency image, because it makes regions except for edges over-blurred.
	//	If a pure edge image is used, at regions near edges the joint/cross bilateral filer may under-blur the ambient(input) image, rather may sharpen the image.
	//	Similarly, at regions except for edges the filter may over-blur the ambient(input) image.

	const bool use_high_freq_image = true;

	if (!high_freq_filename.empty())
	{
		std::cout << "Load an image with high-frequency components '" << high_freq_filename << "'... " << std::flush;

		std::ifstream ppm_in2(high_freq_filename.c_str(), std::ios::binary);

		ppm_in2.get(magic_number[0]);
		ppm_in2.get(magic_number[1]);

		if (magic_number != std::string("P6"))
		{
			std::cerr << "error: unrecognized file format\n" << high_freq_filename << " is not a PPM file.\n" << std::endl;
			return;
		}

		unsigned width2, height2, bpp2;
		ppm_in2 >> width2 >> height2 >> bpp2;
		if (255 != bpp)
		{
			std::cerr << "error: unsupported maximum value (" << bpp << ")\n" << "It must be 255." << std::endl;
			return;
		}

		if ((width2 != width) || (height2 != height))
		{
			std::cerr << "error: image size don't match" << std::endl
				<< "input image: " << width << " x " << height << std::endl
				<< "high-frequency image:  " << width2 << " x " << height2 << std::endl;
			return;
		}

		ppm_in.get(ch);  // Trailing white space.
		high_freq_img = cv::Mat::zeros(height, width, CV_8UC1);
		for (unsigned y = 0; y < height; ++y)
		{
			for (unsigned x = 0; x < width; ++x)
			{
				ppm_in2.get(r);
				ppm_in2.get(g);
				ppm_in2.get(b);

				const unsigned char R = static_cast<unsigned char>(r);
				const unsigned char G = static_cast<unsigned char>(g);
				const unsigned char B = static_cast<unsigned char>(b);

				high_freq_image(x, y) = (20.0 * R + 40.0 * G + 1.0 * B) / (61.0 * 255.0); 

				high_freq_img.at<unsigned char>(y, x) = (unsigned char)cvRound(high_freq_image(x, y) * 255.0);
			}
		}

		ppm_in2.close();
	}
	else
	{
		const cv::Mat input_img = cv::imread(input_filename);
		cv::Mat gray_img(input_img.size(), CV_8UC1);
		//high_freq_img = cv::Mat::zeros(input_img.size(), CV_8UC1);

		for (unsigned y = 0; y < height; ++y)
		{
			for (unsigned x = 0; x < width; ++x)
			{
				const cv::Vec3b rgb(input_img.at<cv::Vec3b>(y, x));
				gray_img.at<unsigned char>(y, x) = (unsigned char)cvRound((20.0 * rgb[0] + 40.0 * rgb[1] + 1.0 * rgb[2]) / 61.0);
			}
		}

		{
			cv::Mat tmp;

			// smoothing.
			local::smooth_image(gray_img, tmp);

			// detect edge.
			local::detect_edge(tmp, high_freq_img);

			for (unsigned y = 0; y < height; ++y)
				for (unsigned x = 0; x < width; ++x)
					high_freq_image(x, y) = high_freq_img.at<unsigned char>(y, x) / 255.0;
		}
	}
#else
	// FIXME [implement] >> use high-frequency image.

	const bool use_high_freq_image = false;
#endif

	std::cout << "Done" << std::endl;

	std::cout << "sigma_s = " << sigma_s << std::endl;
	std::cout << "sigma_r = " << sigma_r << std::endl;

	//---------------------------------------------------------------

	std::cout << "Filter the image... " << std::endl;

	image_type filtered_image(width, height);
	if (use_high_freq_image)
	{
		//Image_filter::linear_BF(input_image, high_freq_image, sigma_s, sigma_r, sigma_s, sigma_r, false, &filtered_image);  // compile-time error.
		Image_filter::fast_LBF(input_image, high_freq_image, sigma_s, sigma_r, false, &filtered_image, &filtered_image);  // fast linear bilateral filter.
	}
	else
	{
		FFT::Support_3D::set_fftw_flags(FFTW_ESTIMATE);
		//Image_filter::linear_BF(input_image, sigma_s, sigma_r, &filtered_image);
		Image_filter::fast_LBF(input_image, sigma_s, sigma_r, &filtered_image);  // fast linear bilateral filter.
	}

	std::cout << "Filtering done" << std::endl;

	//---------------------------------------------------------------

	{
		std::cout << "Write the output image '" << output_filename << "'... " << std::flush;

		std::ofstream ppm_out(output_filename.c_str(), std::ios::binary);

		ppm_out << "P6";
		ppm_out << ' ';
		ppm_out << width;
		ppm_out << ' ';
		ppm_out << height;
		ppm_out << ' ';
		ppm_out << "255";
		ppm_out << std::endl;

		for (unsigned y = 0; y < height; ++y)
		{
			for (unsigned x = 0; x < width; ++x)
			{
				const double R = filtered_image(x, y) * 255.0;
				const double G = filtered_image(x, y) * 255.0;
				const double B = filtered_image(x, y) * 255.0;

				const char r = static_cast<unsigned char>(Math_tools::clamp(0.0, 255.0, R));
				const char g = static_cast<unsigned char>(Math_tools::clamp(0.0, 255.0, G));
				const char b = static_cast<unsigned char>(Math_tools::clamp(0.0, 255.0, B));

				ppm_out << r << g << b;
			}
		}

		ppm_out.flush();
		ppm_out.close();

		std::cout << "Done" << std::endl;
	}

	//---------------------------------------------------------------

	{
		const cv::Mat input_img = cv::imread(input_filename);
		const cv::Mat output_img = cv::imread(output_filename);

		cv::imshow("cross bilateral filter - input", input_img);
		cv::imshow("cross bilateral filter - output", output_img);
		if (!high_freq_img.empty())
			cv::imshow("cross bilateral filter - high-frequency", high_freq_img);

		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}

}  // namespace my_fast_bilateral_filter
