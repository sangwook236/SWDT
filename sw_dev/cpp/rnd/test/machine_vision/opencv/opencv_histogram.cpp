//#include "stdafx.h"
//#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>


namespace {
namespace local {

void draw_histogram_1D(const cv::MatND &hist, const int binCount, const double maxVal, const int binWidth, const int maxHeight, cv::Mat &histImg)
{
#if 0
	for (int i = 0; i < binCount; ++i)
	{
		const float binVal(hist.at<float>(i));
		const int binHeight(cvRound(binVal * maxHeight / maxVal));
		cv::rectangle(
			histImg,
			cv::Point(i*binWidth, maxHeight), cv::Point((i+1)*binWidth - 1, maxHeight - binHeight),
			binVal > maxVal ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 255),
			CV_FILLED
		);
	}
#else
	const float *binPtr = (const float *)hist.data;
	for (int i = 0; i < binCount; ++i, ++binPtr)
	{
		const int binHeight(cvRound(*binPtr * maxHeight / maxVal));
		cv::rectangle(
			histImg,
			cv::Point(i*binWidth, maxHeight), cv::Point((i+1)*binWidth - 1, maxHeight - binHeight),
			*binPtr > maxVal ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 255),
			CV_FILLED
		);
	}
#endif
}

void draw_histogram_2D(const cv::MatND &hist, const int horzBinCount, const int vertBinCount, const double maxVal, const int horzBinSize, const int vertBinSize, cv::Mat &histImg)
{
#if 0
	for (int v = 0; v < vertBinCount; ++v)
		for (int h = 0; h < horzBinCount; ++h)
		{
			const float binVal(hist.at<float>(v, h));
			cv::rectangle(
				histImg,
				cv::Point(h*horzBinSize, v*vertBinSize), cv::Point((h+1)*horzBinSize - 1, (v+1)*vertBinSize - 1),
				binVal > maxVal ? CV_RGB(255, 0, 0) : cv::Scalar::all(cvRound(binVal * 255.0 / maxVal)),
				CV_FILLED
			);
		}
#else
	const float *binPtr = (const float *)hist.data;
	for (int v = 0; v < vertBinCount; ++v)
		for (int h = 0; h < horzBinCount; ++h, ++binPtr)
		{
			const int intensity();
			cv::rectangle(
				histImg,
				cv::Point(h*horzBinSize, v*vertBinSize), cv::Point((h+1)*horzBinSize - 1, (v+1)*vertBinSize - 1),
				*binPtr > maxVal ? cv::Scalar(CV_RGB(255, 0, 0)) : cv::Scalar::all(cvRound(*binPtr * 255.0 / maxVal)),
				CV_FILLED
			);
		}
#endif
}

// the function normalizes the histogram bins by scaling them, such that the sum of the bins becomes equal to factor
void normalize_histogram(cv::MatND &hist, const double factor)
{
#if 0
	// FIXME [modify] >>
	cvNormalizeHist(&(CvHistogram)hist, factor);
#else
	const cv::Scalar sums(cv::sum(hist));

	const double eps = 1.0e-20;
	if (std::fabs(sums[0]) < eps) return;

	cv::Mat tmp(hist);
	tmp.convertTo(hist, -1, factor / sums[0], 0.0);
#endif
}

void histogram_1D()
{
	const std::string imgName("machine_vision_data\\opencv\\lena_gray.bmp");

	//
	const cv::Mat &src = cv::imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
	if (src.empty())
	{
		std::cerr << "image cannot be loaded !!!" << std::endl;
		return;
	}

	// calculate histogram
	const int dims = 1;
	const int bins = 256;
	const int histSize[] = { bins };
	const float range[] = { 0, 256 };
	const float *ranges[] = { range };
	const int channels[] = { 0 };

#if 1
	cv::MatND hist;  // return type: CV_32FC1, 1-dim (rows = bins, cols = 1)
	cv::calcHist(
		&src, 1, channels, cv::Mat(), // do not use mask
		hist, dims, histSize, ranges,
		true, // the histogram is uniform
		false
	);
#else
	const int interval = (range1[1] - range1[0]) / bins;

	cv::MatND hist(cv::MatND::zeros(bins, 1, CV_32F));
	const unsigned char *imgPtr = (unsigned char *)src.data;
	float *binPtr = (float *)hist.data;
	for (int i = 0; i < src.rows * src.cols; ++i, ++imgPtr)
	{
		//const int idx = (imgPtr[channels[0]] - range[0]) / interval;
		//++binPtr[idx];
		++(binPtr[*imgPtr]);
	}
#endif

	// normalize histogram
	const double factor = 1000.0;
	normalize_histogram(hist, factor);

	//
#if 1
	double maxVal = 0.0;
	cv::minMaxLoc(hist, NULL, &maxVal, NULL, NULL);
#else
	const double maxVal = factor * 0.05;
#endif

	// draw 1-D histogram
	const int bin_width = 1, max_height = 100;
	cv::Mat histImg(cv::Mat::zeros(max_height, bins*bin_width, CV_8UC3));
	draw_histogram_1D(hist, bins, maxVal, bin_width, max_height, histImg);

	//
	const std::string windowName("histogram 1D");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowName, histImg);

	cv::waitKey();

	cv::destroyWindow(windowName);
}

void histogram_2D()
{
	const std::string imgName("machine_vision_data\\opencv\\lena_rgb.bmp");

	//
	const cv::Mat &src = cv::imread(imgName, CV_LOAD_IMAGE_COLOR);
	if (src.empty())
	{
		std::cerr << "image cannot be loaded !!!" << std::endl;
		return;
	}

	cv::Mat hsv;
	cv::cvtColor(src, hsv, CV_BGR2HSV);

	// calculate histogram
	const int dims = 2;
	// let's quantize the hue to 30 levels and the saturation to 32 levels
	const int bins1 = 30, bins2 = 32;
	const int histSize[] = { bins1, bins2 };
	// hue varies from 0 to 179, see cvtColor
	const float range1[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
	const float range2[] = { 0, 256 };
	const float *ranges[] = { range1, range2 };
	// we compute the histogram from the 0-th and 1-st channels
	const int channels[] = { 0, 1 };

#if 1
	cv::MatND hist;  // return type: CV_32FC1, 2-dim (rows = bins1, cols = bins2)
	cv::calcHist(
		&hsv, 1, channels, cv::Mat(), // do not use mask
		hist, dims, histSize, ranges,
		true, // the histogram is uniform
		false
	);
#else
	const int interval1 = (range1[1] - range1[0]) / bins1;
	const int interval2 = (range2[1] - range2[0]) / bins2;

	cv::MatND hist(cv::MatND::zeros(bins1, bins2, CV_32F));
	const unsigned char *imgPtr = (unsigned char *)hsv.data;
	float *binPtr = (float *)hist.data;
	for (int i = 0; i < hsv.rows * hsv.cols; ++i, imgPtr += 3)
	{
		const int idx1 = (imgPtr[channels[0]] - range1[0]) / interval1;
		const int idx2 = (imgPtr[channels[1]] - range2[0]) / interval2;
		++*(binPtr + idx1 * hist.cols + idx2);
	}
#endif

	// normalize histogram
	const double factor = 1000.0;
	normalize_histogram(hist, factor);

	//
#if 0
	double maxVal = 0.0;
	cv::minMaxLoc(hist, NULL, &maxVal, NULL, NULL);
#else
	const double maxVal = factor * 0.05;
#endif

	// draw 2-D histogram
	const int hscale = 10, sscale = 10;
	cv::Mat histImg(cv::Mat::zeros(bins2*sscale, bins1*hscale, CV_8UC3));
	draw_histogram_2D(hist, bins1, bins2, maxVal, hscale, sscale, histImg);

	//
	const std::string windowName("histogram 2D");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowName, histImg);

	cv::waitKey();

	cv::destroyWindow(windowName);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void histogram()
{
	//local::histogram_1D();
	local::histogram_2D();
}

}  // namespace my_opencv
