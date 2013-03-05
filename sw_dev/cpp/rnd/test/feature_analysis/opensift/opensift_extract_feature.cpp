//#include "stdafx.h"
#include <opensift/sift.h>
#include <opensift/imgfeatures.h>
#include <opensift/utils.h>

#include <opencv/highgui.h>

#include <stdio.h>

namespace {
namespace local {

/******************************** Globals ************************************/

char *img_file_name = "./feature_analysis_data/sift/beaver.png";
char *out_file_name = "./feature_analysis_data/sift/beaver.sift";
//char *img_file_name = "./feature_analysis_data/sift/marker_pen_2.bmp";
//char *out_file_name = "./feature_analysis_data/sift/marker_pen_2.sift";
char *out_img_name = NULL;
const int display = 1;
const int intvls = SIFT_INTVLS;
const double sigma = SIFT_SIGMA;
const double contr_thr = SIFT_CONTR_THR;
const int curv_thr = SIFT_CURV_THR;
const int img_dbl = SIFT_IMG_DBL;
const int descr_width = SIFT_DESCR_WIDTH;
const int descr_hist_bins = SIFT_DESCR_HIST_BINS;

}  // namespace local
}  // unnamed namespace

namespace my_opensift {

// [ref] ${OPENSIFT_HOME}/src/siftfeat.c
void extract_feature()
{
	IplImage *img;
	struct feature *features;
	int n = 0;

	fprintf(stderr, "Finding SIFT features...\n");
	img = cvLoadImage(local::img_file_name, 1);
	if (!img)
	{
		fprintf(stderr, "unable to load image from %s", local::img_file_name);
		exit(1);
	}
	n = _sift_features(img, &features, local::intvls, local::sigma, local::contr_thr, local::curv_thr, local::img_dbl, local::descr_width, local::descr_hist_bins);
	fprintf(stderr, "Found %d features.\n", n);

	if (local::display)
	{
		draw_features(img, features, n);
		cvNamedWindow(local::img_file_name, 1);
		cvShowImage(local::img_file_name, img);
		cvWaitKey(0);
	}

	if (NULL != local::out_file_name)
		export_features(local::out_file_name, features, n);

	if (NULL != local::out_img_name)
		cvSaveImage(local::out_img_name, img);
}

}  // namespace my_opensift
