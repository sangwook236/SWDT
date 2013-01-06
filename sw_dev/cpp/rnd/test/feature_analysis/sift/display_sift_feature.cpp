/*
Displays image features from a file on an image

Copyright (C) 2006  Rob Hess <hess@eecs.oregonstate.edu>

@version 1.1.1-20070913
*/

#include "stdafx.h"
#include <sift/imgfeatures.h>
#include <sift/utils.h>

#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include <iostream>

namespace {

/******************************** Globals ************************************/

char *feat_file = ".\\sift_data\\beaver.sift";
char *img_file = ".\\sift_data\\beaver.png";
//char *feat_file = ".\\sift_data\\marker_pen_2.sift";
//char *img_file = ".\\sift_data\\marker_pen_2.bmp";
const int feat_type = FEATURE_LOWE;

}  // unnamed namespace


void display_sift_feature()
{
	IplImage *img;
	struct feature *feat;
	char *name;
	int n;

	img = cvLoadImage(img_file, 1);
	if (!img)
		fatal_error("unable to load image from %s", img_file);
	n = import_features(feat_file, feat_type, &feat);
	if (-1 == n)
		fatal_error("unable to import features from %s", feat_file);
	name = feat_file;

	draw_features(img, feat, n);
	cvNamedWindow(name, 1);
	cvShowImage(name, img);
	cvWaitKey(0);
}