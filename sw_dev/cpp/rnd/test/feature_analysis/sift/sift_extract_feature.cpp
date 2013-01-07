/*
This program detects image features using SIFT keypoints. For more info,
refer to:

Lowe, D. Distinctive image features from scale-invariant keypoints.
International Journal of Computer Vision, 60, 2 (2004), pp.91--110.

Copyright (C) 2006  Rob Hess <hess@eecs.oregonstate.edu>

Note: The SIFT algorithm is patented in the United States and cannot be
used in commercial products without a license from the University of
British Columbia.  For more information, refer to the file LICENSE.ubc
that accompanied this distribution.

Version: 1.1.1-20070913
*/

//#include "stdafx.h"
#include <sift/sift.h>
#include <sift/imgfeatures.h>
#include <sift/utils.h>

#include <opencv/highgui.h>

#include <stdio.h>

namespace {
namespace local {

/******************************** Globals ************************************/

char *img_file_name = ".\\feature_analysis_data\\sift\\beaver.png";
char *out_file_name = ".\\feature_analysis_data\\sift\\beaver.sift";
//char *img_file_name = ".\\feature_analysis_data\\sift\\marker_pen_2.bmp";
//char *out_file_name = ".\\feature_analysis_data\\sift\\marker_pen_2.sift";
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

namespace sift {

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

}  // namespace sift
