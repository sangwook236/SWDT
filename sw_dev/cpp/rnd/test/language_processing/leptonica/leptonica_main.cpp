//#include "stdafx.h"
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_leptonica {
	
// REF [site] >> https://stackoverflow.com/questions/39293922/convert-between-opencv-mat-and-leptonica-pix
Pix * mat8ToPix(const cv::Mat &mat8)
{
	Pix *pix = pixCreate(mat8.size().width, mat8.size().height, 8);
	for (int y = 0; y < mat8.rows; ++y)
		for (int x = 0; x < mat8.cols; ++x)
			pixSetPixel(pix, x, y, (l_uint32)mat8.at<uchar>(y, x));
	return pix;
}

// REF [site] >> https://stackoverflow.com/questions/39293922/convert-between-opencv-mat-and-leptonica-pix
cv::Mat pix8ToMat(const Pix *pix8)
{
	cv::Mat mat(cv::Size(pix8->w, pix8->h), CV_8UC1);
	uint32_t *line = pix8->data;
	for (uint32_t y = 0; y < pix8->h; ++y)
	{
		for (uint32_t x = 0; x < pix8->w; ++x)
			mat.at<uchar>(y, x) = GET_DATA_BYTE(line, x);
		line += pix8->wpl;
	}
	return mat;
}

cv::Mat pix32ToMat(const Pix *pix32)
{
	cv::Mat mat(cv::Size(pix32->w, pix32->h), CV_8UC4);
	uint32_t *line = pix32->data;
	for (uint32_t y = 0; y < pix32->h; ++y)
	{
		for (uint32_t x = 0; x < pix32->w; ++x)
		{
			const uint8_t *ptr = (uint8_t *)&GET_DATA_FOUR_BYTES(line, x);
			//mat.at<cv::Vec4b>(y, x) = cv::Vec4b(ptr);  // No conversion.
			mat.at<cv::Vec4b>(y, x) = cv::Vec4b(*(ptr + 3), *(ptr + 2), *(ptr + 1), *(ptr));  // Reverse order.
			//mat.at<cv::Vec4b>(y, x) = cv::Vec4b(*(ptr + 2), *(ptr + 1), *(ptr + 0), *(ptr + 3));  // RGBA -> BGRA.
			//mat.at<cv::Vec4b>(y, x) = cv::Vec4b(*(ptr), *(ptr + 3), *(ptr + 2), *(ptr + 1));  // ARGB -> ABGR.
		}
		line += pix32->wpl;
	}
	return mat;
}

}  // namespace my_leptonica

int leptonica_main(int argc, char *argv[])
{
	throw std::runtime_error("Not yet implemented");

	return 0;
}

