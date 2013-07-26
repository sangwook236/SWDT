//#include "stdafx.h"
#define CHRONO
#include "../fast_bilateral_filter_lib/geom.h"
#include "../fast_bilateral_filter_lib/fast_lbf.h"
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
	const std::string input_filename("./signal_processing_data/fast_bilateral_filter/building.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/dome.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/dragon_hires.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/dragon_lores.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/flower.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/housecorner_hires.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/housecorner_lores.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/rock.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/swamp.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/synthetic.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/temple.ppm");
	//const std::string input_filename("./signal_processing_data/fast_bilateral_filter/turtle.ppm");
	
	const std::string output_filename("./signal_processing_data/fast_bilateral_filter/cross_bf_output.ppm");
	
	//const std::string edge_filename("./signal_processing_data/fast_bilateral_filter/edge.ppm");
	const std::string edge_filename;
	
	const double sigma_s = 16;  // space sigma.
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

	image_type image(width, height);

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

			image(x, y) = (20.0 * R + 40.0 * G + 1.0 * B) / (61.0 * 255.0); 
		}
	}

	ppm_in.close();

	std::cout << "Done" << std::endl;

	//---------------------------------------------------------------

	image_type edge(width, height);
	cv::Mat edge_img;
	if (!edge_filename.empty())
	{
		std::cout << "Load the edge image '" << edge_filename << "'... " << std::flush;

		std::ifstream ppm_in2(edge_filename.c_str(), std::ios::binary);

		ppm_in2.get(magic_number[0]);
		ppm_in2.get(magic_number[1]);

		if (magic_number != std::string("P6"))
		{
			std::cerr << "error: unrecognized file format\n" << edge_filename << " is not a PPM file.\n" << std::endl;
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
				<< "input: " << width << " x " << height << std::endl
				<< "edge:  " << width2 << " x " << height2 << std::endl;
			return;
		}

		ppm_in.get(ch);  // Trailing white space.
		edge_img = cv::Mat::zeros(height, width, CV_8UC1);
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

				edge(x, y) = (20.0 * R + 40.0 * G + 1.0 * B) / (61.0 * 255.0); 

				edge_img.at<unsigned char>(y, x) = (unsigned char)cvRound(edge(x, y) * 255.0);
			}
		}

		ppm_in2.close();
	}
	else
	{
		const cv::Mat input_img = cv::imread(input_filename);
		cv::Mat gray_img(input_img.size(), CV_8UC1);
		//edge_img = cv::Mat::zeros(input_img.size(), CV_8UC1);

		for (unsigned y = 0; y < height; ++y)
		{
			for (unsigned x = 0; x < width; ++x)
			{
				const cv::Vec3b rgb(input_img.at<cv::Vec3b>(y, x));
				gray_img.at<unsigned char>(y, x) = (unsigned char)cvRound((20.0 * rgb[0] + 40.0 * rgb[1] + 1.0 * rgb[2]) / 61.0);
			}
		}

		// edge detection.
#if 0
		// down-scale and up-scale the image to filter out the noise
		cv::Mat blurred_img;
		cv::pyrDown(gray_img, blurred_img);
		cv::pyrUp(blurred_img, edge_img);
#elif 1
		const int ksize = 3;
		cv::blur(gray_img, edge_img, cv::Size(ksize, ksize));
#else
		edge_img = gray_img.clone();
#endif

		// run the edge detector on grayscale
		const int lowerEdgeThreshold = 30, upperEdgeThreshold = 50;
		const int apertureSize = 3;  // aperture size for the Sobel() operator.
		const bool useL2 = true;  // if true, use L2 norm. otherwise, use L1 norm (faster).
		cv::Canny(edge_img, edge_img, lowerEdgeThreshold, upperEdgeThreshold, apertureSize, useL2);

		for (unsigned y = 0; y < height; ++y)
			for (unsigned x = 0; x < width; ++x)
				edge(x, y) = edge_img.at<unsigned char>(y, x) / 255.0;
	}

	std::cout << "Done" << std::endl;

	std::cout << "sigma_s    = " << sigma_s << std::endl;
	std::cout << "sigma_r    = " << sigma_r << std::endl;

	//---------------------------------------------------------------

	std::cout << "Filter the image... " << std::endl;

	image_type filtered_image(width, height);
	Image_filter::fast_LBF(image, edge, sigma_s, sigma_r, false, &filtered_image, &filtered_image);

	std::cout << "Filtering done" << std::endl;

	//---------------------------------------------------------------

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

	//---------------------------------------------------------------

	{
		const cv::Mat input_image = cv::imread(input_filename);
		const cv::Mat output_image = cv::imread(output_filename);

		cv::imshow("cross bilateral filter - input", input_image);
		cv::imshow("cross bilateral filter - output", output_image);
		if (!edge_img.empty())
			cv::imshow("cross bilateral filter - edge", edge_img);

		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}

}  // namespace my_fast_bilateral_filter
