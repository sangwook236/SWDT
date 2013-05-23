#include <vl/slic.h>
#include <vl/generic.h>
#include <vl/stringop.h>
#include <vl/pgm.h>
#include <opencv2/opencv.hpp>
#include <iostream>


namespace {
namespace local {
	
bool read_pgm(const std::string &name, vl_uint8 *& data, VlPgmImage &pim, const bool verbose)
{
	FILE *in = fopen(name.c_str(), "rb");
	if (!in)
	{
		std::cerr << "could not open '" << name.c_str() << "' for reading." << std::endl;
		return false;
	}
	// read source image header
	vl_bool err = vl_pgm_extract_head(in, &pim);
	if (err)
	{
		std::cerr << "PGM header corrputed." << std::endl;
		return false;
	}

	if (verbose)
		std::cout << "SLIC:   image is " << pim. width << " by " << pim. height << " pixels" << std::endl;

	// allocate buffer
	data = new vl_uint8 [vl_pgm_get_npixels(&pim) * vl_pgm_get_bpp(&pim)];
	if (!data)
	{
		std::cerr << "could not allocate enough memory." << std::endl;
		return false;
	}

	// read PGM
	err = vl_pgm_extract_data(in, &pim, data);
	if (err)
	{
		std::cerr << "PGM body corrputed." << std::endl;
		return false;
	}

	return true;
}

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

// simple linear iterative clustering (SLIC)
// superpixel extraction (segmentation) method based on a local version of k-means
void slic()
{
    // read image data
#if 0
	const std::string input_filename = "./machine_vision_data/vlfeat/box.pgm";
	const bool verbose = true;

	vl_uint8 *data = NULL;
	VlPgmImage pim;
	if (!local::read_pgm(input_filename, data, pim, verbose))
		return;

	//
	const vl_size img_width = pim.width;
	const vl_size img_height = pim.height;
	const vl_size img_numChannels = 1;

	float *image = new float [img_width * img_height];
	for (vl_size i = 0; i < img_width * img_height; ++i)
		image[i] = data[i];

	if (data)
	{
		delete [] data;
		data = NULL;
	}
#elif 1
	const std::string input_filename = "./machine_vision_data/opencv/fruits.jpg";

	const cv::Mat input_img = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);

	//
	const vl_size img_width = input_img.cols;
	const vl_size img_height = input_img.rows;
	const vl_size img_numChannels = 1;

	float *image = new float [img_width * img_height * img_numChannels];
	{
		for (size_t i = 0; i < img_width * img_height * img_numChannels; ++i)
			image[i] = input_img.data[i];
	}
#else
	const std::string input_filename = "./machine_vision_data/opencv/fruits.jpg";

	const cv::Mat input_img = cv::imread(input_filename);

	//
	const vl_size img_width = input_img.cols;
	const vl_size img_height = input_img.rows;
	const vl_size img_numChannels = 3;

	float *image = new float [img_width * img_height * img_numChannels];
	{
		for (size_t i = 0; i < img_width * img_height * img_numChannels; ++i)
			image[i] = input_img.data[i];
	}
#endif

	//
	vl_uint32 *segmentation = new vl_uint32 [img_width * img_height];
	const vl_size regionSize = 100;
	const float regularization = 1.0f;
	const vl_size minRegionSize = 10;
	vl_slic_segment(segmentation, image, img_width, img_height, img_numChannels, regionSize, regularization, minRegionSize);

	//
	const vl_uint32 maxLabel = *std::max_element(segmentation, segmentation + img_width * img_height);

	// visualize
	cv::Mat result_img(img_height, img_width, CV_32FC1);
	{
		vl_size idx = 0;
		for (int r = 0; r < result_img.rows; ++r)
				for (int c = 0; c < result_img.cols; ++c, ++idx)
					result_img.at<float>(r, c) = segmentation[idx] / (float)maxLabel;
	}

	cv::imshow("SLIC result", result_img);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// clean-up
	if (segmentation)
	{
		delete [] segmentation;
		segmentation = NULL;
	}
	if (image)
	{
		delete [] image;
		image = NULL;
	}
}

}  // namespace my_vlfeat
