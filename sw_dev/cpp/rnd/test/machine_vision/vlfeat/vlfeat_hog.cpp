#include "generic-driver.h"

#include <vl/hog.h>
#include <vl/stringop.h>
#include <vl/pgm.h>
#include <vl/hog.h>
#include <vl/getopt_long.h>

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <string>
#include <iostream>


namespace {
namespace local {

bool werr(const vl_bool err, const std::string &name)
{
	if (err == VL_ERR_OVERFLOW)
	{
		std::cerr << "output file name too long." << std::endl;
		return false;
	}
	else if (err)
	{
		std::cerr << "could not open '" << name << "' for writing." << std::endl;
		return false;
	}

	return true;
}

bool read_pgm(const std::string &name, vl_uint8 *&data, VlPgmImage &pim, const bool verbose)
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
		std::cout << "hog:   image is " << pim. width << " by " << pim. height << " pixels" << std::endl;

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

// [ref]
//	"Histograms of oriented gradients for human detection", N. Dalal and B. Triggs. CVPR, 2005.
//	"Object detection with discriminatively trained part based models", P. F. Felzenszwalb, R. B. Grishick, D. McAllester, and D. Ramanan. PAMI, 2010.

void hog()
{
	const std::string input_filename = "./data/machine_vision/vlfeat/box.pgm";

	// algorithm parameters.
	vl_bool err = VL_ERR_OK;
	int exit_code = 0;
	const bool verbose = true;

	VlFileMeta frm = { 0, "%.frame", VL_PROT_ASCII, "", 0 };
	VlFileMeta piv = { 0, "%.hog",  VL_PROT_ASCII, "", 0 };
	VlFileMeta met = { 0, "%.meta",  VL_PROT_ASCII, "", 0 };

	// get basenmae from filename.
    char basename[1024];
	vl_size q = vl_string_basename(basename, sizeof(basename), input_filename.c_str(), 1) ;
	err = (q >= sizeof(basename));
	if (err)
	{
		std::cerr << "basename of '" << input_filename.c_str() << "' is too long" << std::endl;
		err = VL_ERR_OVERFLOW;
		return;
	}

	if (verbose)
	{
		std::cout << "hog: processing " << input_filename.c_str() << std::endl;
		std::cout << "hog:    basename is " << basename << std::endl;
	}

	// open output files.
	err = vl_file_meta_open(&piv, basename, "w");  if (!local::werr(err, piv.name)) return;
	err = vl_file_meta_open(&frm, basename, "w");  if (!local::werr(err, frm.name)) return;
	err = vl_file_meta_open(&met, basename, "w");  if (!local::werr(err, met.name)) return;

	if (verbose)
	{
		if (piv.active) std::cout << "hog:  writing seeds  to " << piv.name << std::endl;
		if (frm.active) std::cout << "hog:  writing frames to " << frm.name << std::endl;
		if (met.active) std::cout << "hog:  writing meta   to " << met.name << std::endl;
	}

    // read image data.
    vl_uint8 *data_uint8 = NULL;
	VlPgmImage pim;
	if (!local::read_pgm(input_filename, data_uint8, pim, verbose))
		return;

	float *data_float = new float [vl_pgm_get_npixels(&pim) * vl_pgm_get_bpp(&pim)];
	for (vl_size i = 0; i < vl_pgm_get_npixels(&pim) * vl_pgm_get_bpp(&pim); ++i)
		data_float[i] = (float)data_uint8[i] / (float)pim.max_value;

	// process data.
	const vl_size numOrientations = 9;  // number of distinguished orientations.
	const vl_size cellSize = 8;  // size of a HOG cell.
	const vl_bool imageTransposed = VL_FALSE;  // wether images are transposed (column major).

#if 1
	// the original Dalal-Triggs variant (with 2¡¿2 square HOG blocks for normalization).
	// Dalal-Triggs works with undirected gradients only and does not do any compression, for a total of 36 dimension.
	VlHog *hog = vl_hog_new(VlHogVariantDalalTriggs, numOrientations, imageTransposed);
#else
	// the UoCTTI variant.
	// the UoCTTI variant computes both directed and undirected gradients as well as a four dimensional texture-energy feature, but projects the result down to 31 dimensions.
	// [ref] "Object Detection with Discriminatively Trained Part-Based Models", P. Felzenszwalb, R. Girshick, D. McAllester, & D. Ramanan, TPAMI, 2010.
	VlHog *hog = vl_hog_new(VlHogVariantUoctti, numOrientations, imageTransposed);
#endif

#if 1
	const vl_size numChannels = 1;
	vl_hog_put_image(hog, data_float, pim.width, pim.height, numChannels, cellSize);
#else
	// FIXME [implement] >>

	float *modulus = ;  // image gradient modulus.
	float *angle = ;  // image gradient angle.
	const vl_bool directed = true;  // wrap the gradient angles at 2pi (directed) or pi (undirected).
	vl_hog_put_polar_field(hog, modulus, angle, directed, pim.width, pim.height, cellSize);
#endif

	//
	const vl_size hogWidth = vl_hog_get_width(hog);
	const vl_size hogHeight = vl_hog_get_height(hog);
	const vl_size hogDimension = vl_hog_get_dimension(hog);

	float *hogFeatures = (float *)vl_malloc(hogWidth * hogHeight * hogDimension * sizeof(float));
	vl_hog_extract(hog, hogFeatures);

	// display output.
	const vl_size glyphSize = vl_hog_get_glyph_size(hog);
	const vl_size hogImageHeight = glyphSize * hogHeight;
	const vl_size hogImageWidth = glyphSize * hogWidth;
	float *hogImage = (float *)vl_malloc(sizeof(float) * hogImageWidth * hogImageHeight);
	vl_hog_render(hog, hogImage, hogFeatures, hogWidth, hogHeight);

	// FIXME [delete] >>
	const float maxPix1 = *std::max_element(hogFeatures, hogFeatures + hogWidth * hogHeight * hogDimension);
	const float maxPix2 = *std::max_element(hogImage, hogImage + hogImageHeight * hogImageWidth);

	// FIXME [fix] >>
	//  -. This is working well in Linux, but not Windows.
	//	-. maxPix2 is zero. why?

	{
		cv::Mat input_img((int)pim.height, (int)pim.width, CV_32FC1, data_float);
#if 1
		cv::Mat result_img((int)hogImageHeight, (int)hogImageWidth, CV_32FC1, hogImage);
#else
		cv::Mat result_img = cv::Mat::zeros((int)hogImageHeight, (int)hogImageWidth, CV_32FC1);
		for (int r = 0, idx = 0; r < result_img.rows; ++r)
		{
			float *ptr = result_img.ptr<float>(r);
			for (int c = 0; c < result_img.cols; ++c, ++idx)
				ptr[c] = hogImage[idx];
		}
#endif

		const float maxPix = *std::max_element(hogImage, hogImage + hogImageHeight * hogImageWidth);
		for (int r = 0; r < result_img.rows; ++r)
			for (int c = 0; c < result_img.cols; ++c)
				result_img.at<float>(r, c) = result_img.at<float>(r, c) / maxPix;

		cv::resize(result_img, result_img, cv::Size(input_img.cols, input_img.rows));

		cv::imshow("HOG - input", input_img);
		cv::imshow("HOG - result", result_img);

		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	if (hogImage)
	{
		vl_free(hogImage);
		hogImage = NULL;
	}

	//
	if (hogFeatures)
	{
		vl_free(hogFeatures);
		hogFeatures = NULL;
    }

    if (hog)
	{
		vl_hog_delete(hog);
		hog = NULL;
    }

	// release image data.
	if (data_uint8)
	{
		delete [] data_uint8;
		data_uint8 = NULL;
	}
	if (data_float)
	{
		delete [] data_float;
		data_float = NULL;
	}

    vl_file_meta_close(&frm);
    vl_file_meta_close(&piv);
    vl_file_meta_close(&met);
}

}  // namespace my_vlfeat
