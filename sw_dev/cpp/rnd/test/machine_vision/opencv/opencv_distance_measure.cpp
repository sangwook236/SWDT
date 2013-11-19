//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <fstream>


namespace my_opencv {

void normalize_histogram(cv::MatND &hist, const double factor);

}  // namespace my_opencv

namespace {
namespace local {

void histogram_comparison()
{
	const double arr01[] = { 2.5906700e+00, 1.2953400e+00, 6.4766800e+00, 0.0000000e+00, 0.0000000e+00, 3.8860100e+00, 2.5906700e+00, 0.0000000e+00, 0.0000000e+00, 3.8860100e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 2.5906700e+00, 3.8860100e+00, 0.0000000e+00, 3.8860100e+00, 2.5906700e+00, 1.2953400e+00, 5.1813500e+00, 1.2953400e+00, 2.5906700e+00, 1.2953400e+00, 5.1813500e+00, 0.0000000e+00, 1.2953400e+00, 1.2953400e+00, 0.0000000e+00, 1.2953400e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 1.2953400e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 2.5906700e+00, 0.0000000e+00, 1.2953400e+00, 2.5906700e+00, 0.0000000e+00, 2.5906700e+00, 2.5906700e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 1.2953400e+00, 1.2953400e+00, 1.2953400e+00, 0.0000000e+00, 2.5906700e+00, 0.0000000e+00, 1.2953400e+00, 1.2953400e+00, 2.5906700e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 2.5906700e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 0.0000000e+00, 1.2953400e+00, 1.2953400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2953400e+00, 1.2953400e+00, 5.1813500e+00, 2.5906700e+00, 5.1813500e+00, 7.7720200e+00, 6.4766800e+00, 2.5906700e+00, 5.1813500e+00, 2.5906700e+00, 6.4766800e+00, 6.4766800e+00, 5.1813500e+00, 2.5906700e+00, 7.7720200e+00, 9.0673600e+00, 6.4766800e+00, 2.0725400e+01, 1.2953400e+01, 1.5544000e+01, 1.1658000e+01, 2.0725400e+01, 1.1658000e+01, 1.2953400e+01, 1.8134700e+01, 1.6839400e+01, 1.5544000e+01, 2.4611400e+01, 1.4248700e+01, 1.4248700e+01, 1.5544000e+01, 1.6839400e+01, 1.0362700e+01, 7.7720200e+00, 9.0673600e+00, 3.8860100e+00, 5.1813500e+00, 7.7720200e+00, 2.5906700e+00, 2.5906700e+00, 3.8860100e+00 };
	const double arr02[] = { 4.2949200e+00, 2.1474600e+00, 1.0737300e+00, 3.5791000e-01, 4.2949200e+00, 1.0737300e+00, 1.0737300e+00, 7.1582000e-01, 0.0000000e+00, 1.0737300e+00, 3.5791000e-01, 7.1582000e-01, 1.4316400e+00, 3.5791000e-01, 1.7895500e+00, 7.1582000e-01, 3.5791000e-01, 3.5791000e-01, 7.1582000e-01, 3.5791000e-01, 1.0737300e+00, 0.0000000e+00, 7.1582000e-01, 3.5791000e-01, 0.0000000e+00, 7.1582000e-01, 3.5791000e-01, 0.0000000e+00, 0.0000000e+00, 3.5791000e-01, 0.0000000e+00, 3.5791000e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.5791000e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.5791000e-01, 0.0000000e+00, 3.5791000e-01, 0.0000000e+00, 7.1582000e-01, 0.0000000e+00, 3.5791000e-01, 0.0000000e+00, 1.0737300e+00, 3.5791000e-01, 7.1582000e-01, 7.1582000e-01, 3.5791000e-01, 1.4316400e+00, 3.5791000e-01, 1.0737300e+00, 1.4316400e+00, 1.0737300e+00, 1.7895500e+00, 1.4316400e+00, 1.7895500e+00, 1.7895500e+00, 2.5053700e+00, 2.1474600e+00, 2.8632800e+00, 2.5053700e+00, 2.1474600e+00, 2.5053700e+00, 1.7895500e+00, 3.9370100e+00, 2.5053700e+00, 3.2211900e+00, 3.5791000e+00, 3.5791000e+00, 4.6528300e+00, 5.3686500e+00, 4.2949200e+00, 2.1474600e+00, 1.4316400e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.5791000e-01, 7.1582000e-01, 3.9370100e+00, 2.1474600e+00, 4.2949200e+00, 2.1474600e+00, 1.4316400e+00, 0.0000000e+00, 3.5791000e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.5791000e-01, 0.0000000e+00, 0.0000000e+00, 3.5791000e-01, 0.0000000e+00, 7.1582000e-01, 3.5791000e-01, 1.0737300e+00, 7.1582000e-01, 0.0000000e+00, 3.5791000e-01, 3.5791000e-01, 3.5791000e-01, 1.0737300e+00, 3.5791000e-01, 1.4316400e+00, 3.5791000e-01, 1.0737300e+00, 3.5791000e-01, 1.0737300e+00, 7.1582000e-01, 1.0737300e+00, 1.0737300e+00, 1.0737300e+00, 3.2211900e+00, 1.4316400e+00, 1.4316400e+00, 1.0737300e+00, 1.0737300e+00, 1.7895500e+00, 3.2211900e+00, 1.4316400e+00, 1.7895500e+00, 2.1474600e+00, 2.1474600e+00, 3.2211900e+00, 2.1474600e+00, 2.1474600e+00, 2.8632800e+00, 3.9370100e+00, 2.8632800e+00, 4.2949200e+00, 6.4423800e+00, 2.2548300e+01, 1.2168900e+01, 6.4423800e+00, 3.9370100e+00, 6.4423800e+00, 4.6528300e+00, 4.6528300e+00, 6.4423800e+00, 3.5791000e+00, 5.7265600e+00, 6.0844700e+00, 4.2949200e+00, 6.4423800e+00, 6.4423800e+00, 6.8002900e+00, 8.2319300e+00, 1.1095200e+01, 1.3600600e+01, 1.5032200e+01, 3.0064400e+01, 4.4022900e+01, 2.8632800e+01, 2.0042900e+01, 1.9685000e+01, 5.0107400e+00, 6.8002900e+00, 8.2319300e+00, 8.5898400e+00 };

	cv::Mat histo1, histo2;
	cv::Mat(1, 360, CV_64FC1, (void *)arr01).convertTo(histo1, CV_32FC1, 1.0, 0.0);
	cv::Mat(1, 360, CV_64FC1, (void *)arr02).convertTo(histo2, CV_32FC1, 1.0, 0.0);

	std::cout << "distances between two histograms:" << std::endl;
	double dist;

	{
		boost::timer::auto_cpu_timer timer;
		// correlation: better match has higher score - perfect match = 1.0, total mismatch = -1.0.
		dist = cv::compareHist(histo1, histo2, CV_COMP_CORREL);
	}
	std::cout << "\tCorrelation:   " << dist << std::endl;

	{
		boost::timer::auto_cpu_timer timer;
		// chi-square: better match has lower score - perfect match = 0.0, mismatch > 0.0.
		dist = cv::compareHist(histo1, histo2, CV_COMP_CHISQR);
	}
	std::cout << "\tChi-Square:    " << dist << std::endl;

	{
		boost::timer::auto_cpu_timer timer;
		// histogram intersection: better match has higher score - perfect match = 1.0, total mismatch = 0.0 if two histograms are normalized to 1.
		dist = cv::compareHist(histo1, histo2, CV_COMP_INTERSECT);
	}
	std::cout << "\tIntersection:  " << dist << std::endl;

	{
		boost::timer::auto_cpu_timer timer;
		// Bhattacharyya: better match has lower score - perfect match = 0.0, total mismatch = 1.0.
		dist = cv::compareHist(histo1, histo2, CV_COMP_BHATTACHARYYA);
	}
	std::cout << "\tBhattacharyya: " << dist << std::endl;
}

// [ref] histogram_2D() in opencv_histogram().
void compute_histogram(const cv::Mat &src, cv::Mat &histo, const int h_bins, const int s_bins)
{
	// create images.
	cv::Mat hsv;
	cv::cvtColor(src, hsv, CV_BGR2HSV);

	// calculate histogram.
	const int dims = 2;
	// let's quantize the hue to 30 levels and the saturation to 32 levels.
	const int bins1 = h_bins, bins2 = s_bins;
	const int histSize[] = { bins1, bins2 };
	// hue varies from 0 to 179, see cvtColor.
	const float range1[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to 255 (pure spectrum color).
	const float range2[] = { 0, 256 };
	const float *ranges[] = { range1, range2 };
	// we compute the histogram from the 0th and 1st channels.
	const int channels[] = { 0, 1 };

#if 1
	//cv::MatND histo;  // return type: CV_32FC1, 2-dim (rows = bins1, cols = bins2).
	cv::calcHist(
		&hsv, 1, channels, cv::Mat(), // do not use mask.
		histo, dims, histSize, ranges,
		true, // the histogram is uniform.
		false
	);
#else
	const int interval1 = (range1[1] - range1[0]) / bins1;
	const int interval2 = (range2[1] - range2[0]) / bins2;

	//cv::MatND histo(cv::MatND::zeros(bins1, bins2, CV_32F));
	histo = cv::MatND::zeros(bins1, bins2, CV_32F);
	const unsigned char *imgPtr = (unsigned char *)hsv.data;
	float *binPtr = (float *)histo.data;
	for (int i = 0; i < hsv.rows * hsv.cols; ++i, imgPtr += 3)
	{
		const int idx1 = (imgPtr[channels[0]] - range1[0]) / interval1;
		const int idx2 = (imgPtr[channels[1]] - range2[0]) / interval2;
		++*(binPtr + idx1 * histo.cols + idx2);
	}
#endif

	// normalize histogram.
	const double factor = 1.0;
	my_opencv::normalize_histogram(histo, factor);
}

void earth_movers_distance()
{
#if 1
	const std::string img1_filename("./data/machine_vision/opencv/lena_rgb.bmp");
	const std::string img2_filename("./data/machine_vision/opencv/lena_rgb.bmp");
	//const std::string img2_filename("./data/machine_vision/opencv/lena_gray.bmp");
#elif 0
	const std::string img1_filename("./data/machine_vision/teddy-imL.png");
	//const std::string img2_filename("./data/machine_vision/teddy-imL.png");
	const std::string img2_filename("./data/machine_vision/teddy-imR.png");
#endif

	// load images.
	const cv::Mat &img1 = cv::imread(img1_filename, CV_LOAD_IMAGE_COLOR);
	if (img1.empty())
	{
		std::cerr << "image file not found: " << img1_filename << std::endl;
		return;
	}
	const cv::Mat &img2 = cv::imread(img2_filename, CV_LOAD_IMAGE_COLOR);
	if (img2.empty())
	{
		std::cerr << "image file not found: " << img2_filename << std::endl;
		return;
	}

	// compute histograms.
	const int h_bins = 30;
	const int s_bins = 32;

	cv::Mat histo1, histo2;
	compute_histogram(img1, histo1, h_bins, s_bins);
	compute_histogram(img2, histo2, h_bins, s_bins);

	// create matrices to store signature in.
	// (histogram's size) x (histogram's dim. + 1) floating-point matrix.
	//	in case of 1D, (histogram's size) = (bin size), (histogram's dim. + 1) = 1 count(bin value) + 1 coords = 2.
	//	in case of 2D, (histogram's size) = (row's bin size) x (col's bin size), (histogram's dim. + 1) = 1 count(bin value) + 2 coords = 3.
#if 0
	// fill signatures for the two histograms.
	const int num_rows = h_bins * s_bins;
	cv::Mat sig1(num_rows, 3, CV_32FC1, cv::Scalar::all(0.0f)), sig2(num_rows, 3, CV_32FC1, cv::Scalar::all(0.0f));
	for (int h = 0; h < h_bins; ++h)
		for (int s = 0; s < s_bins; ++s)
		{
			const float weight1 = histo1.at<float>(h, s);
			if (weight1 > 0.0)
			{
				sig1.at<float>(h * s_bins + s, 0) = weight1;  // bin value (weight).
				sig1.at<float>(h * s_bins + s, 1) = (float)h;  // coord 1.
				sig1.at<float>(h * s_bins + s, 2) = (float)s;  // coord 2.
			}

			const float weight2 = histo2.at<float>(h, s);
			if (weight2 > 0.0)
			{
				sig2.at<float>(h * s_bins + s, 0) = weight2;  // bin value (weight).
				sig2.at<float>(h * s_bins + s, 1) = (float)h;  // coord 1.
				sig2.at<float>(h * s_bins + s, 2) = (float)s;  // coord 2.
			}
		}

	float dist;
	{
		boost::timer::auto_cpu_timer timer;
		// better match has lower score - perfect match = 0.0, total mismatch = 1.0(?).
		dist = cv::EMD(sig1, sig2, CV_DIST_L2);
	}
	std::cout << "earth mover's distance between two histograms: " << dist << std::endl;
#elif 1
	// fill signatures for the two histograms (partial matching).
	cv::Mat sig1(h_bins * s_bins, 3, CV_32FC1, cv::Scalar::all(0.0f));
	for (int h = 0; h < h_bins; ++h)
		for (int s = 0; s < s_bins; ++s)
		{
			const float weight1 = histo1.at<float>(h, s);
			if (weight1 > 0.0)
			{
				sig1.at<float>(h * s_bins + s, 0) = weight1;  // bin value (weight).
				sig1.at<float>(h * s_bins + s, 1) = (float)h;  // coord 1.
				sig1.at<float>(h * s_bins + s, 2) = (float)s;  // coord 2.

#if 0
				// for debugging.
				std::cout << '(' << h << ", " << s << ") : " << weight1 << std::endl;
#endif
			}
		}
	const int s_bin_size = 15;
	for (int kk = 0; kk <= s_bins - s_bin_size; ++kk)
	{
		const int s_bin_start = kk, s_bin_end = kk + s_bin_size - 1;
		// re-normalize histogram.
#if defined(__GNUC__)
        cv::Mat colMat = histo2.colRange(s_bin_start, s_bin_end + 1);
		my_opencv::normalize_histogram(colMat, 1.0);
#else
		my_opencv::normalize_histogram(histo2.colRange(s_bin_start, s_bin_end + 1), 1.0);
#endif

		cv::Mat sig2(h_bins * s_bin_size, 3, CV_32FC1, cv::Scalar::all(0.0f));
		for (int h = 0; h < h_bins; ++h)
			for (int s = s_bin_start; s <= s_bin_end && s < s_bins; ++s)
			{
				const float weight2 = histo2.at<float>(h, s);
				if (weight2 > 0.0)
				{
					sig2.at<float>(h * s_bin_size + s - s_bin_start, 0) = weight2;  // bin value (weight).
					sig2.at<float>(h * s_bin_size + s - s_bin_start, 1) = (float)h;  // coord 1.
					sig2.at<float>(h * s_bin_size + s - s_bin_start, 2) = (float)s;  // coord 2.
					//sig2.at<float>(h * s_bin_size + s - s_bin_start, 2) = (float)(s - s_bin_start);  // coord 2.

#if 0
					// for debugging.
					std::cout << '(' << h << ", " << s << ") : " << weight2 << std::endl;
#endif
				}
			}

		float dist;
		{
			boost::timer::auto_cpu_timer timer;
			// better match has lower score - perfect match = 0.0, total mismatch = 1.0(?).
			dist = cv::EMD(sig1, sig2, CV_DIST_L2);
		}
		std::cout << "earth mover's distance between two histograms: " << dist << std::endl;
	}
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void distance_measure()
{
	//local::histogram_comparison();

	// earth mover's distance.
	local::earth_movers_distance();
}

}  // namespace my_opencv
