//include "stdafx.h"
#include "../shadows_lib/ChromacityShadRem.h"
#include "../shadows_lib/GeometryShadRem.h"
#include "../shadows_lib/LrTextureShadRem.h"
#include "../shadows_lib/PhysicalShadRem.h"
#include "../shadows_lib/SrTextureShadRem.h"
#include <opencv2/opencv.hpp>
#include <iostream>


namespace {
namespace local {

// [ref] ${SHADOWS_HOME}/src/main.cpp.
void simple_example()
{
	// load frame, background and foreground.
	const cv::Mat &frame = cv::imread("./data/object_detection/shadows/frame.bmp");
	const cv::Mat &bg = cv::imread("./data/object_detection/shadows/bg.bmp");
	const cv::Mat &fg = cv::imread("./data/object_detection/shadows/fg.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	// create shadow removers.
	ChromacityShadRem chr;
	PhysicalShadRem phy;
	GeometryShadRem geo;
	SrTextureShadRem srTex;
	LrTextureShadRem lrTex;

	// matrices to store the masks after shadow removal.
	cv::Mat chrMask, phyMask, geoMask, srTexMask, lrTexMask;

	// remove shadows.
	chr.removeShadows(frame, fg, bg, chrMask);
	phy.removeShadows(frame, fg, bg, phyMask);
	geo.removeShadows(frame, fg, bg, geoMask);
	srTex.removeShadows(frame, fg, bg, srTexMask);
	lrTex.removeShadows(frame, fg, bg, lrTexMask);

	// show results.
	cv::imshow("frame", frame);
	cv::imshow("bg", bg);
	cv::imshow("fg", fg);
	cv::imshow("chr", chrMask);
	cv::imshow("phy", phyMask);
	cv::imshow("geo", geoMask);
	cv::imshow("srTex", srTexMask);
	cv::imshow("lrTex", lrTexMask);

	cv::waitKey();
}

}  // namespace local
}  // unnamed namespace

namespace my_shadows {

}  // namespace my_shadows

int shadows_main(int argc, char *argv[])
{
	local::simple_example();

	return 0;
}
