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
	{
		std::cerr << "file not found: " << input_filename << std::endl;
		return;
	}

	//
	const vl_size img_width = pim.width;
	const vl_size img_height = pim.height;
	const vl_size img_numChannels = 1;

	float *image = new float [img_width * img_height];
	for (vl_size i = 0; i < img_width * img_height; ++i)
		image[i] = data[i] / 255.0;

	if (data)
	{
		delete [] data;
		data = NULL;
	}
#elif 1
	const std::string input_filename = "./machine_vision_data/vlfeat/slic_image.jpg";
	//const std::string input_filename = "./machine_vision_data/opencv/fruits.jpg";

	const cv::Mat input_img = cv::imread(input_filename, CV_LOAD_IMAGE_COLOR);
	//const cv::Mat input_img = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (input_img.empty())
	{
		std::cerr << "file not found: " << input_filename << std::endl;
		return;
	}

	//
	const vl_size img_width = input_img.cols;
	const vl_size img_height = input_img.rows;
	const vl_size img_numChannels = input_img.channels();

	// channels * width * height + width * row + col
	std::vector<float> image(img_width * img_height * img_numChannels, 0.0);
	{
		if (1 == img_numChannels)
		{
			for (vl_size r = 0; r < img_height; ++r)
				for (vl_size c = 0; c < img_width; ++c)
					image[img_width * r + c] = (float)input_img.at<unsigned char>(r, c);
		}
		else if (3 == img_numChannels)
		{
			for (vl_size r = 0; r < img_height; ++r)
				for (vl_size c = 0; c < img_width; ++c)
				{
					const cv::Vec3b &pix = input_img.at<cv::Vec3b>(r, c);
					for (vl_size ch = 0; ch < img_numChannels; ++ch)
						image[ch * img_width * img_height + img_width * r + c] = (float)pix[ch];
				}
		}
		else
		{
			std::cerr << "the number of image's channels is improper ..." << std::endl;
			return;
		}
	}
#endif

	//
	const vl_size regionSize = 30;  // nominal size of the regions.
	const float regularization = 10.0f;  // trade-off between appearance and spatial terms.
	const vl_size minRegionSize = (vl_size)cvRound((regionSize / 6.0) * (regionSize / 6.0));  // minimum size of a segment.

	std::vector<vl_uint32> segmentation(img_width * img_height, 0);
	vl_slic_segment(&segmentation[0], &image[0], img_width, img_height, img_numChannels, regionSize, regularization, minRegionSize);

	// visualize
	cv::Mat result_img(input_img.size(), input_img.type());
	{
#if 0
		const vl_uint32 maxLabel = *std::max_element(segmentation.begin(), segmentation.end());

		if (1 == result_img.channels())
		{
			vl_size idx = 0;
			for (int r = 0; r < result_img.rows; ++r)
				for (int c = 0; c < result_img.cols; ++c, ++idx)
					result_img.at<unsigned char>(r, c) = (unsigned char)cvRound(segmentation[idx] * 255 / float(maxLabel));
		}
		else if (3 == result_img.channels())
		{
			vl_size idx = 0;
			for (int r = 0; r < result_img.rows; ++r)
				for (int c = 0; c < result_img.cols; ++c, ++idx)
				{
					const unsigned char val = (unsigned char)cvRound(segmentation[idx] * 255 / float(maxLabel));
					result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(val, val, val);
				}
		}
		else
		{
			std::cerr << "the number of image's channels is improper ..." << std::endl;
			return;
		}
#else
		// draw boundary.
		if (1 == result_img.channels())
		{
			vl_size idx = 0;
			for (int r = 0; r < result_img.rows; ++r)
				for (int c = 0; c < result_img.cols; ++c, ++idx)
				{
					const int lbl = segmentation[idx];
					if (r - 1 >= 0 && lbl != segmentation[(r - 1) * result_img.cols + c])
						result_img.at<unsigned char>(r, c) = 255;
					else if (c - 1 >= 0 && lbl != segmentation[r * result_img.cols + (c - 1)])
						result_img.at<unsigned char>(r, c) = 255;
/*
					else if (r + 1 < result_img.rows && lbl != segmentation[(r + 1) * result_img.cols + c])
						result_img.at<unsigned char>(r, c) = 255;
					else if (c + 1 < result_img.cols && lbl != segmentation[r * result_img.cols + (c + 1)])
						result_img.at<unsigned char>(r, c) = 255;
*/
					else
						result_img.at<unsigned char>(r, c) = 0;
				}
		}
		else if (3 == result_img.channels())
		{
			vl_size idx = 0;
			for (int r = 0; r < result_img.rows; ++r)
				for (int c = 0; c < result_img.cols; ++c, ++idx)
				{
					const int lbl = segmentation[idx];
					if (r - 1 >= 0 && lbl != segmentation[(r - 1) * result_img.cols + c])
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
					else if (c - 1 >= 0 && lbl != segmentation[r * result_img.cols + (c - 1)])
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
/*
					else if (r + 1 < result_img.rows && lbl != segmentation[(r + 1) * result_img.cols + c])
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
					else if (c + 1 < result_img.cols && lbl != segmentation[r * result_img.cols + (c + 1)])
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
*/
					else
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				}
		}
		else
		{
			std::cerr << "the number of image's channels is improper ..." << std::endl;
			return;
		}
#endif
	}

	cv::imshow("SLIC result", result_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

}  // namespace my_vlfeat
