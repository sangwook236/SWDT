//#include "stdafx.h"
#include <opensift/imgfeatures.h>
#include <opensift/utils.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opensift {

// [ref] ${OPENSIFT_HOME}/src/dspfeat.c
void display_feature()
{
#if 1
    const std::string feat_file("./data/face_analysis/sift/beaver.sift");
    const std::string img_file("./data/face_analysis/sift/beaver.png");
#elif 0
    const std::string feat_file("./data/face_analysis/sift/marker_pen_2.sift");
    const std::string img_file("./data/face_analysis/sift/marker_pen_2.bmp");
#endif
    const int feat_type = FEATURE_LOWE;

	IplImage *img = cvLoadImage(img_file.c_str(), 1);
	if (!img)
		fatal_error((char *)"unable to load image from %s", img_file.c_str());

	struct feature *feat;
	const int n = import_features((char *)feat_file.c_str(), feat_type, &feat);
	if (-1 == n)
		fatal_error((char *)"unable to import features from %s", feat_file.c_str());

	const char *name = feat_file.c_str();

	draw_features(img, feat, n);
	cvNamedWindow(name, 1);
	cvShowImage(name, img);
	cvWaitKey(0);
}

}  // namespace my_opensift
