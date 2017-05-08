//#include "stdafx.h"
#include "../quirc_lib/quirc.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>


//#define __USE_COLOR_IMAGE_ 1


namespace {
namespace local {

void bgr_to_luma(const cv::Mat &bgrImg, const int dst_pitch, uint8_t *dst)
{
	int y;

	for (y = 0; y < bgrImg.rows; ++y)
	{
		uint8_t *gray = dst + y * dst_pitch;
		for (int i = 0; i < bgrImg.cols; ++i)
		{
			// ITU-R colorspace assumed.
			const cv::Vec3b bgr(bgrImg.at<cv::Vec3b>(y, i));
			const int sum = int(bgr[2]) * 59 + int(bgr[1]) * 150 + int(bgr[0]) * 29;

			*(gray++) = sum >> 8;
		}
	}
}

void gray_to_luma(const cv::Mat &grayImg, const int dst_pitch, uint8_t *dst)
{
	int y;

	for (y = 0; y < grayImg.rows; ++y)
	{
		uint8_t *gray = dst + y * dst_pitch;
		for (int i = 0; i < grayImg.cols; ++i)
			*(gray++) = grayImg.at<unsigned char>(y, i);
	}
}

/*
void print_data(const struct quirc_data *data, struct dthash *dt)
{
	if (dthash_seen(dt, data))
		return;

	std::cout << "==> " << data->payload << std::endl;
	std::cout << "\tVer: " << data->version << ", ECC: " << "MLHQ"[data->ecc_level] << ", Mask: " << data->mask << ", Type: " << data->data_type << std::endl;
}
*/

void draw_qr(cv::Mat &img, struct quirc *qr, struct dthash *dt = nullptr)
{
	const int count = quirc_count(qr);
	std::cout << "Number of detected QR codes = " << count << std::endl;

	cv::RNG &rng = cv::theRNG();
	for (int i = 0; i < count; ++i)
	{
		int xc = 0;
		int yc = 0;

		struct quirc_code code;
		quirc_extract(qr, i, &code);

		const cv::Scalar color = count > 1 ? CV_RGB(rng(256), rng(256), rng(256)) : CV_RGB(255, 0, 0);
		for (int j = 0; j < 4; ++j)
		{
			struct quirc_point *a = &code.corners[j];
			struct quirc_point *b = &code.corners[(j + 1) % 4];

			xc += a->x;
			yc += a->y;

			cv::line(img, cv::Point(a->x, a->y), cv::Point(b->x, b->y), color, 1, cv::LINE_AA);
		}

		xc /= 4;
		yc /= 4;

		//
		std::cout << "\tCode size: " << code.size << " cells" << std::endl;

		struct quirc_data data;
		const quirc_decode_error_t err = quirc_decode(&code, &data);
		if (err)
		{
			std::cerr << "\tError: " << quirc_strerror(err) << std::endl;
		}
		else
		{
			std::cout << "\tPayload: " << (char *)data.payload << std::endl;
			//print_data(&data, dt);

			std::cout << "\tVer: " << data.version << ", ECC: " << "MLHQ"[data.ecc_level] << ", Mask: " << data.mask << ", Type: " << data.data_type << std::endl;
		}
	}
}

// REF [file] >> ${QUIRC_HOME}/demo/demo.c.
void simple_example()
{
	//const std::string img_filename("./data/machine_vision/qr_code_north.png");
	//const std::string img_filename("./data/machine_vision/qr_code_south.png");
	//const std::string img_filename("./data/machine_vision/qr_code_east.png");
	//const std::string img_filename("./data/machine_vision/qr_code_west.png");
	//const std::string img_filename("./data/machine_vision/qr_code_tilt.png");
	//const std::string img_filename("./data/machine_vision/qr_code_1.png");
	//const std::string img_filename("./data/machine_vision/qr_code_2.png");  // Detect, but not correct.
	//const std::string img_filename("./data/machine_vision/qr_code_3.png");  // Detect, but not correct.
	//const std::string img_filename("./data/machine_vision/qr_code_4.png");
	const std::string img_filename("./data/machine_vision/qr_code_5.png");  // Detect only two.
	//const std::string img_filename("./data/machine_vision/qr_code_6.png");  // Incorrect.

#if defined(__USE_COLOR_IMAGE_)
	const cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
#else
	const cv::Mat img(cv::imread(img_filename, cv::IMREAD_GRAYSCALE));
#endif
	if (img.empty())
	{
		std::cerr << "ERR: Failed to open image: " << img_filename << std::endl;
		return;
}

	struct quirc *qr = quirc_new();
	if (!qr)
	{
		std::cerr << "Couldn't allocate QR decoder" << std::endl;
		return;
	}

	if (quirc_resize(qr, img.cols, img.rows) < 0)
	{
		std::cerr << "Couldn't allocate QR buffer" << std::endl;
		quirc_destroy(qr);
		return;
	}

	uint8_t *dst = quirc_begin(qr, NULL, NULL);
#if defined(__USE_COLOR_IMAGE_)
	bgr_to_luma(img, img.cols, dst);
#else
	gray_to_luma(img, img.cols, dst);
#endif
	quirc_end(qr);

	// Show the result.
	cv::Mat rgb;
#if defined(__USE_COLOR_IMAGE_)
	img.copyTo(rgb);
#else
	cv::cvtColor(img, rgb, cv::COLOR_GRAY2BGR);
#endif
	draw_qr(rgb, qr);

	cv::imshow("QR code - Input image", img);
	cv::imshow("QR code - Result image", rgb);

	cv::waitKey();
	cv::destroyAllWindows();

	quirc_destroy(qr);
}

}  // namespace local
}  // unnamed namespace

namespace my_quirc {

}  // namespace my_quirc

// REF [site] >> https://github.com/dlbeer/quirc
int quirc_main(int argc, char *argv[])
{
	try
	{
		cv::theRNG();

		local::simple_example();
	}
	catch (const cv::Exception& ex)
	{
		//std::cout << "OpenCV exception caught: " << ex.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(ex.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << ex.err << std::endl
			<< "\tline:        " << ex.line << std::endl
			<< "\tfunction:    " << ex.func << std::endl
			<< "\tfile:        " << ex.file << std::endl;

		return 1;
	}

    return 0;
}
