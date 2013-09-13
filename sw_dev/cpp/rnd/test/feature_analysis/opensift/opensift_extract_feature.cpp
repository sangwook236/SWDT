//#include "stdafx.h"
#include <opensift/sift.h>
#include <opensift/imgfeatures.h>
#include <opensift/utils.h>
#include <opencv/highgui.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opensift {

// [ref] ${OPENSIFT_HOME}/src/siftfeat.c
void extract_feature()
{
#if 1
    const std::string img_file_name("./data/feature_analysis/sift/beaver.png");
    const std::string out_file_name("./data/feature_analysis/sift/beaver.sift");
#elif 0
    const std::string img_file_name("./data/feature_analysis/sift/marker_pen_2.bmp");
    const std::string out_file_name("./data/feature_analysis/sift/marker_pen_2.sift");
#endif
    const std::string out_img_name;

    const int display = 1;
    const int intvls = SIFT_INTVLS;
    const double sigma = SIFT_SIGMA;
    const double contr_thr = SIFT_CONTR_THR;
    const int curv_thr = SIFT_CURV_THR;
    const int img_dbl = SIFT_IMG_DBL;
    const int descr_width = SIFT_DESCR_WIDTH;
    const int descr_hist_bins = SIFT_DESCR_HIST_BINS;

	std::cout << "finding SIFT features..." << std::endl;
	IplImage *img = cvLoadImage(img_file_name.c_str(), 1);
	if (!img)
	{
		std::cout <<"unable to load image from " << img_file_name << std::endl;
		return;
	}

	struct feature *features;
	const int n = _sift_features(img, &features, intvls, sigma, contr_thr, curv_thr, img_dbl, descr_width, descr_hist_bins);
	std::cout << "found " << n << " features." << std::endl;

	if (display)
	{
		draw_features(img, features, n);
		cvNamedWindow(img_file_name.c_str(), 1);
		cvShowImage(img_file_name.c_str(), img);
		cvWaitKey(0);
	}

	if (!out_file_name.empty())
		export_features((char *)out_file_name.c_str(), features, n);

	if (!out_img_name.empty())
		cvSaveImage(out_img_name.c_str(), img);
}

}  // namespace my_opensift
