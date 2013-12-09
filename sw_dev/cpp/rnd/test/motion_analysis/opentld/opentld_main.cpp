//include "stdafx.h"
#include <opentld/TLDUtil.h>
#include <opentld/TLD.h>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

void simple_test()
{

#if 1
	const std::string avi_filename("./data/motion_analysis/TownCentreXVID.avi");
	cv::VideoCapture capture(avi_filename);

	const cv::Rect initBB(725, 239, 813 - 725, 426 - 239);
#else
	const int camId = -1;
	cv::VideoCapture capture(camId);
#endif
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	boost::scoped_ptr<tld::TLD> tld(new tld::TLD());

	std::cout << "start capturing ... " << std::endl;
	cv::Mat frame, image, grey;
	tld::ForegroundDetector *fg;
	bool isInitialized = false;
	while (true)
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			//break;
			continue;
		}

#if 1
		if (image.empty()) image = frame.clone();
		else frame.copyTo(image);
#else
		image = frame;
#endif
		if (image.empty()) continue;

#if 0
		cv::flip(image, image, 0);  // flip vertically (around x-axis)
#elif 0
		cv::flip(image, image, 1);  // flip horizontally (around y-axis)
#elif 0
		cv::flip(image, image, -1);  // flip vertically & horizontally (around both axes)
#endif

	    cv::cvtColor(image, grey, CV_BGR2GRAY);

		if (!isInitialized)
		{
			tld->detectorCascade->imgWidth = grey.cols;
			tld->detectorCascade->imgHeight = grey.rows;
			tld->detectorCascade->imgWidthStep = grey.step;

			//tld->learning = ;
			//tld->learningEnabled = ;
			//tld->alternating = ;
			//tld->writeToFile(modelExportFilePath);
			//tld->readFromFile(modelImportFilePath);

			//const int initBB[] = { 0, 0, 100, 100 };
			//const cv::Rect bb = tld::tldArrayToRect(initBB);
			//tld->selectObject(grey, (cv::Rect *)&bb);
			tld->selectObject(grey, (cv::Rect *)&initBB);

			isInitialized = true;
		}

		tld->processImage(image);

		if (NULL != tld->currBB)
			std::cout << tld->currBB->x << ", " << tld->currBB->y << ", " << tld->currBB->width << ", " << tld->currBB->height << ", " << tld->currConf << std::endl;

		for (std::size_t i = 0; i < tld->detectorCascade->detectionResult->fgList->size(); ++i)
		{
			const cv::Rect &r = tld->detectorCascade->detectionResult->fgList->at(i);
			cv::rectangle(image, r.tl(), r.br(), CV_RGB(255, 255, 255), 1, 8, 0);
		}

		fg = tld->detectorCascade->foregroundDetector;
		if (fg->bgImg.empty())
			fg->bgImg = grey.clone();
		else
			fg->bgImg.release();

		//
		cv::imshow("OpenTLD - result", image);

		const int key = cv::waitKey(1);
		switch (key)
		{
		case 'q':
			break;
		}
	}
	std::cout << "end capturing ... " << std::endl;

	cv::destroyAllWindows();

	// Clean-up.
	tld->release();
}

}  // namespace local
}  // unnamed namespace

namespace my_opentld {

}  // namespace my_opentld

int opentld_main(int argc, char *argv[])
{
	try
	{
		cv::theRNG();

		local::simple_test();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		return 1;
	}

	return 0;
}
