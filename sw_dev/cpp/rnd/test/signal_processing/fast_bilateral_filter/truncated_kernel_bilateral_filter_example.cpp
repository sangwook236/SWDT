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

// [ref] ${FAST_BILATERAL_FILTER_HOME}/TRUNCATED_KERNEL_BF/truncated_kernel_bf.cpp
void truncated_kernel_bilateral_filter_example()
{
	typedef Array_2D<double> image_type;

#if 0
	if (5 != argc)
	{
		std::cerr << "error: wrong arguments"<< std::endl;
		std::cerr << std::endl;
		std::cerr << "usage: "<<argv[0]<<" input.ppm output.ppm sigma_s sigma_r" << std::endl;
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
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/building.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/dome.ppm");
	const std::string input_filename("./data/signal_processing/fast_bilateral_filter/dragon_hires.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/dragon_lores.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/flower.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/housecorner_hires.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/housecorner_lores.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/rock.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/swamp.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/synthetic.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/temple.ppm");
	//const std::string input_filename("./data/signal_processing/fast_bilateral_filter/turtle.ppm");
	
	const std::string output_filename("./data/signal_processing/fast_bilateral_filter/truncated_kernel_bf_output.ppm");

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

	std::cout << "sigma_s    = " << sigma_s << std::endl;
	std::cout << "sigma_r    = " << sigma_r << std::endl;

	//---------------------------------------------------------------

	std::cout << "Filter the image... " << std::endl;

	image_type filtered_image(width, height);
	Image_filter::fast_LBF(image, image, sigma_s, sigma_r, false, &filtered_image, &filtered_image);  // fast linear bilateral filter.

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

		cv::imshow("truncated kernel bilateral filter - input", input_image);
		cv::imshow("truncated kernel bilateral filter - output", output_image);

		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}

}  // namespace my_fast_bilateral_filter
