#include <vl/lbp.h>
#include <vl/generic.h>
#include <vl/stringop.h>
#include <vl/pgm.h>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <cmath>


namespace {
namespace local {

// [ref] ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp.
void draw_histogram_1D(const cv::MatND &histo, const int binCount, const double maxVal, const int binWidth, const int maxHeight, cv::Mat &histo_img)
{
#if 0
	for (int i = 0; i < binCount; ++i)
	{
		const float binVal(histo.at<float>(i));
		const int binHeight(cvRound(binVal * maxHeight / maxVal));
		cv::rectangle(
			histo_img,
			cv::Point(i*binWidth, maxHeight), cv::Point((i+1)*binWidth - 1, maxHeight - binHeight),
			binVal > maxVal ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 255),
			cv::FILLED
		);
	}
#else
	const float *binPtr = (float *)(histo.data);
	for (int i = 0; i < binCount; ++i, ++binPtr)
	{
		const int binHeight(cvRound(*binPtr * maxHeight / maxVal));
		cv::rectangle(
			histo_img,
			cv::Point(i*binWidth, maxHeight), cv::Point((i+1)*binWidth - 1, maxHeight - binHeight),
			*binPtr > maxVal ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 255),
			cv::FILLED
		);
	}
#endif
}

// [ref] ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp.
void normalize_histogram(cv::MatND &hist, const double factor)
{
#if 0
	// FIXME [modify] >>
	cvNormalizeHist(&(CvHistogram)hist, factor);
#else
	const cv::Scalar sums(cv::sum(hist));

	const double eps = 1.0e-20;
	if (std::fabs(sums[0]) < eps) return;

	//cv::Mat tmp(hist);
	//tmp.convertTo(hist, -1, factor / sums[0], 0.0);
	hist *= factor / sums[0];
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

void lbp()
{
	const std::string input_filename = "../data/machine_vision/vlfeat/box.pgm";
	//const std::string input_filename = "../data/machine_vision/opencv/lena_gray.pgm";

	// read image data.
	float *img_float = NULL;
	VlPgmImage pim;
	if (vl_pgm_read_new_f(input_filename.c_str(), &pim, &img_float))
	{
		std::cerr << "fail to load image, " << input_filename << std::endl;
		return;
	}

	// 3x3 pixel neighbourhoods.
	VlLbp *lbp = vl_lbp_new(VlLbpUniform, false);
	
	// process data.
	const vl_size cell_size = 16;  // size of the LBP cells.

	const vl_size &lbp_dimension = vl_lbp_get_dimension(lbp);
	const vl_size &lbp_rows = (vl_size)std::floor((double)pim.height / cell_size);
	const vl_size &lbp_cols = (vl_size)std::floor((double)pim.width / cell_size);
	float *lbp_features = (float *)vl_malloc(lbp_rows * lbp_cols * lbp_dimension * sizeof(float));
	memset(lbp_features, 0, lbp_rows * lbp_cols * lbp_dimension * sizeof(float));  // necessary.

	// NOTICE [caution] >> 58, but not 256.
	//	[ref] http://www.vlfeat.org/api/lbp_8h.html.
	//std::cout << "the dimension of a LBP feature: " << lbp_dimension << std::endl;

	vl_lbp_process(lbp, lbp_features, img_float, pim.width, pim.height, cell_size);

	// display output.
	{
		// calculate histogram.
		const vl_size lbp_stride = lbp_rows * lbp_cols;
		cv::MatND histo(1, lbp_dimension, CV_32FC1, cv::Scalar::all(0));
		for (vl_size rr = 0, kk = 0; rr < lbp_rows; ++rr)
			for (vl_size cc = 0; cc < lbp_cols; ++cc, ++kk)
				for (vl_size i = 0; i < lbp_dimension; ++i)
					histo.at<float>(i) += lbp_features[i * lbp_stride + kk];  // NOTICE [caution] >>

		// normalize histogram.
		//const double factor = 1000.0;
		//local::normalize_histogram(histo, factor);

		//
#if 1
		double maxVal = 0.0;
		cv::minMaxLoc(histo, NULL, &maxVal, NULL, NULL);
#else
		const double maxVal = factor * 0.05;
#endif

		// draw 1-D histogram.
		const int bin_width = 1, max_height = 100;
		cv::MatND histo_img(cv::Mat::zeros(max_height, lbp_dimension*bin_width, CV_8UC3));
		local::draw_histogram_1D(histo, lbp_dimension, maxVal, bin_width, max_height, histo_img);

		//
		cv::imshow("LBP histogram", histo_img);

		cv::waitKey();
		cv::destroyAllWindows();
	}

	//
	vl_lbp_delete(lbp);

	if (lbp_features)
	{
		vl_free(lbp_features);
		lbp_features = NULL;
    }
	if (img_float)
	{
		vl_free(img_float);
		img_float = NULL;
	}
}

}  // namespace my_vlfeat
