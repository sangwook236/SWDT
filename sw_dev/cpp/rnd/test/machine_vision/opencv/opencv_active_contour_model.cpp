//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// [ref] ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void snake(IplImage *srcImage, IplImage *grayImage);

void active_contour_model()
{
	std::list<std::string> filenames;
	filenames.push_back("./data/machine_vision/opencv/pic1.png");
	filenames.push_back("./data/machine_vision/opencv/pic2.png");
	filenames.push_back("./data/machine_vision/opencv/pic3.png");
	filenames.push_back("./data/machine_vision/opencv/pic4.png");
	filenames.push_back("./data/machine_vision/opencv/pic5.png");
	filenames.push_back("./data/machine_vision/opencv/pic6.png");
	filenames.push_back("./data/machine_vision/opencv/stuff.jpg");
	filenames.push_back("./data/machine_vision/opencv/synthetic_face.png");
	filenames.push_back("./data/machine_vision/opencv/puzzle.png");
	filenames.push_back("./data/machine_vision/opencv/fruits.jpg");
	filenames.push_back("./data/machine_vision/opencv/lena_rgb.bmp");
	filenames.push_back("./data/machine_vision/opencv/hand_01.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_05.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_24.jpg");

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {

		IplImage *srcImage = cvLoadImage(it->c_str());
		if (NULL == srcImage)
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		IplImage *grayImage = NULL;
		if (1 == srcImage->nChannels)
			cvCopy(srcImage, grayImage, NULL);
		else
		{
			grayImage = cvCreateImage(cvGetSize(srcImage), srcImage->depth, 1);
#if defined(__GNUC__)
			if (strcasecmp(srcImage->channelSeq, "RGB") == 0)
#elif defined(_MSC_VER)
			if (_stricmp(srcImage->channelSeq, "RGB") == 0)
#endif
				cvCvtColor(srcImage, grayImage, CV_RGB2GRAY);
#if defined(__GNUC__)
			else if (strcasecmp(srcImage->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
			else if (_stricmp(srcImage->channelSeq, "BGR") == 0)
#endif
				cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);
			else
				assert(false);
			grayImage->origin = srcImage->origin;
		}

		//
		snake(srcImage, grayImage);

		//
		cvShowImage("active contour model", srcImage);

		const unsigned char key = cvWaitKey(0);
		if (27 == key)
			break;

		//
		cvReleaseImage(&grayImage);
		cvReleaseImage(&srcImage);
	}

	cvDestroyAllWindows();
}

}  // namespace my_opencv
