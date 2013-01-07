/*
Displays image features from a file on an image

Copyright (C) 2006  Rob Hess <hess@eecs.oregonstate.edu>

@version 1.1.1-20070913
*/

//#include "stdafx.h"
#include <sift/imgfeatures.h>
#include <sift/utils.h>

#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include <iostream>

namespace {
namespace local {

/******************************** Globals ************************************/

char *feat_file = ".\\feature_analysis_data\\sift\\beaver.sift";
char *img_file = ".\\feature_analysis_data\\sift\\beaver.png";
//char *feat_file = ".\\feature_analysis_data\\sift\\marker_pen_2.sift";
//char *img_file = ".\\feature_analysis_data\\sift\\marker_pen_2.bmp";
const int feat_type = FEATURE_LOWE;

}  // namespace local
}  // unnamed namespace

namespace sift {

void display_feature()
{
	IplImage *img;
	struct feature *feat;
	char *name;
	int n;

	img = cvLoadImage(local::img_file, 1);
	if (!img)
		fatal_error("unable to load image from %s", local::img_file);
	n = import_features(local::feat_file, local::feat_type, &feat);
	if (-1 == n)
		fatal_error("unable to import features from %s", local::feat_file);
	name = local::feat_file;

	draw_features(img, feat, n);
	cvNamedWindow(name, 1);
	cvShowImage(name, img);
	cvWaitKey(0);
}

}  // namespace sift
