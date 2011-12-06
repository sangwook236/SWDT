#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <list>
#include <cassert>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace {

void harris_corner(IplImage *&srcImage, IplImage *grayImage)
{
	const int blockSize = 3;
	const int apertureSize = 7;
	const double k = 0.1;

	IplImage *harrisResponseImage = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_32F, 1);
	harrisResponseImage->origin = srcImage ->origin;

	cvCornerHarris(grayImage, harrisResponseImage, blockSize, apertureSize, k);

	//
	//cvConvertScale(harrisResponseImage, tmp, 255.0, 0.0);  // 32F -> 8U

	cvReleaseImage(&srcImage);
	srcImage = harrisResponseImage;
}

void strong_corner(IplImage *srcImage, IplImage *grayImage)
{
	const double quality = 0.01;
	const double minDistance = 10;
	const int winSize = 10;  // odd number ???

	const int MAX_FEATURE_COUNT = 10000;

	CvPoint2D32f *features = (CvPoint2D32f *)cvAlloc(MAX_FEATURE_COUNT * sizeof(CvPoint2D32f));

	IplImage *eig = cvCreateImage(cvGetSize(grayImage), grayImage->depth, 1);
	IplImage *temp = cvCreateImage(cvGetSize(grayImage), grayImage->depth, 1);

	int featureCount = MAX_FEATURE_COUNT;
	cvGoodFeaturesToTrack(
		grayImage, eig, temp, features, &featureCount,
		quality, minDistance, NULL, 3, 0, 0.04
	);
	cvFindCornerSubPix(
		grayImage, features, featureCount,
		cvSize(winSize, winSize), cvSize(-1, -1),
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03)
	);

	cvReleaseImage(&eig);
	cvReleaseImage(&temp);

	for (int i = 0; i < featureCount; ++i)
	{
		const int r = 3;
		cvCircle(srcImage, cvPoint(cvRound(features[i].x), cvRound(features[i].y)), r, CV_RGB(255,0,0), CV_FILLED, CV_AA, 0);
		cvLine(srcImage, cvPoint(cvRound(features[i].x - r), cvRound(features[i].y)), cvPoint(cvRound(features[i].x + r), cvRound(features[i].y)), CV_RGB(0,255,0), 1, CV_AA, 0);
		cvLine(srcImage, cvPoint(cvRound(features[i].x), cvRound(features[i].y - r)), cvPoint(cvRound(features[i].x), cvRound(features[i].y + r)), CV_RGB(0,255,0), 1, CV_AA, 0);
	}

	cvFree(&features);
}

void surf(IplImage *srcImage, IplImage *grayImage)
{
	const double hessianThreshold = 600;
	const int useExtendedDescriptor = 0;  // 0 means basic descriptors (64 elements each), 1 means extended descriptors (128 elements each)

	CvSeq *keypoints = NULL;
	CvSeq *descriptors = NULL;
	CvMemStorage *storage = cvCreateMemStorage(0);

	cvExtractSURF(grayImage, NULL, &keypoints, &descriptors, storage, cvSURFParams(hessianThreshold, useExtendedDescriptor));

	const int keypointCount = keypoints->total;
	assert(keypointCount == descriptors->total);
	for (int i = 0; i < keypointCount; ++i)
	{
		// SURF point
		const CvSURFPoint *keypoint = (CvSURFPoint *)cvGetSeqElem(keypoints, i);

		// SURF descriptor
		const float *descriptor = (float *)cvGetSeqElem(descriptors, i);

		//cvCircle(srcImage, cvPoint(cvRound(keypoint->pt.x), cvRound(keypoint->pt.y)), 1, CV_RGB(255,0,0), CV_FILLED, CV_AA, 0);
		//cvCircle(srcImage, cvPoint(cvRound(keypoint->pt.x), cvRound(keypoint->pt.y)), 2, CV_RGB(0,0,255), 1, CV_AA, 0);
		const int r = keypoint->size / 10;
		cvCircle(srcImage, cvPoint(cvRound(keypoint->pt.x), cvRound(keypoint->pt.y)), r, CV_RGB(255,0,0), 1, CV_AA, 0);
		cvLine(srcImage, cvPoint(cvRound(keypoint->pt.x), cvRound(keypoint->pt.y)), cvPoint(cvRound(keypoint->pt.x + r * std::cos(keypoint->dir * CV_PI / 180.0)), cvRound(keypoint->pt.y + r * std::sin(keypoint->dir * CV_PI / 180.0))), CV_RGB(0,255,0), 1, CV_AA, 0);
	}

	cvClearMemStorage(storage);
}

void star_keypoint(IplImage *srcImage, IplImage *grayImage)
{
	CvMemStorage *storage = cvCreateMemStorage(0);

	CvSeq *keypoints = cvGetStarKeypoints(grayImage, storage, cvStarDetectorParams(45));

	const int keypointCount = keypoints ? keypoints->total : 0;
	for (int i = 0; i < keypointCount; ++i)
	{
		const CvStarKeypoint *keypoint = (CvStarKeypoint *)cvGetSeqElem(keypoints, i);

		const int r = keypoint->size / 2;
		cvCircle(srcImage, keypoint->pt, r, CV_RGB(255,0,0), 1, CV_AA, 0);
		cvLine(srcImage, cvPoint(keypoint->pt.x + r, keypoint->pt.y + r), cvPoint(keypoint->pt.x - r, keypoint->pt.y - r), CV_RGB(0,255,0), 1, CV_AA, 0);
		cvLine(srcImage, cvPoint(keypoint->pt.x - r, keypoint->pt.y + r), cvPoint(keypoint->pt.x + r, keypoint->pt.y - r), CV_RGB(0,255,0), 1, CV_AA, 0);
	}

	cvClearMemStorage(storage);
}

void mser(IplImage *srcImage, IplImage *grayImage)
{
	const CvScalar colors[] = 
	{
		{{0,0,255}},
		{{0,128,255}},
		{{0,255,255}},
		{{0,255,0}},
		{{255,128,0}},
		{{255,255,0}},
		{{255,0,0}},
		{{255,0,255}},
		{{255,255,255}},
		{{196,255,255}},
		{{255,255,196}}
	};

	const unsigned char bcolors[][3] = 
	{
		{0,0,255},
		{0,128,255},
		{0,255,255},
		{0,255,0},
		{255,128,0},
		{255,255,0},
		{255,0,0},
		{255,0,255},
		{255,255,255}
	};

	IplImage *hsv = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 3);
	cvCvtColor(srcImage, hsv, CV_BGR2YCrCb);

	CvMSERParams params = cvMSERParams();  //cvMSERParams(5, 60, cvRound(0.2 * grayImage->width * grayImage->height), 0.25, 0.2);
	CvMemStorage *storage= cvCreateMemStorage();
	CvSeq *contours = NULL;
	double t = (double)cvGetTickCount();
	cvExtractMSER(hsv, NULL, &contours, storage, params);
	t = cvGetTickCount() - t;

	cvReleaseImage(&hsv);

	std::cout << "MSER extracted " << contours->total << " contours in " << (t/((double)cvGetTickFrequency()*1000.0)) << " ms" << std::endl;

	// draw MSER with different color
	//unsigned char *imgptr = (unsigned char *)srcImage->imageData;
	//for (int i = contours->total - 1; i >= 0; --i)
	//{
	//	CvSeq *seq = *(CvSeq **)cvGetSeqElem(contours, i);
	//	for (int j = 0; j < seq->total; ++j)
	//	{
	//		CvPoint *pt = CV_GET_SEQ_ELEM(CvPoint, seq, j);
	//		imgptr[pt->x*3+pt->y*srcImage->widthStep] = bcolors[i%9][2];
	//		imgptr[pt->x*3+1+pt->y*srcImage->widthStep] = bcolors[i%9][1];
	//		imgptr[pt->x*3+2+pt->y*srcImage->widthStep] = bcolors[i%9][0];
	//	}
	//}

	// find ellipse ( it seems cvFitEllipse2 have error or sth? )
	// FIXME [check] >> there are some errors. have to compare original source (mser_sample.cpp)
	for (int i = 0; i < contours->total; ++i)
	{
		const CvContour *contour = *(CvContour **)cvGetSeqElem(contours, i);
		const CvBox2D box = cvFitEllipse2(contour);
		//box.angle = (float)CV_PI / 2.0f - box.angle;
		
		if (contour->color > 0)
			cvEllipseBox(srcImage, box, colors[9], 2, 8, 0);
		else
			cvEllipseBox(srcImage, box, colors[2], 2, 8, 0);
	}

	cvClearMemStorage(storage);
}

}  // unnamed namespace

void feature_extraction()
{
	std::list<std::string> filenames;

#if 0
	filenames.push_back("opencv_data\\osp_robot_1.jpg");
	filenames.push_back("opencv_data\\osp_robot_2.jpg");
	filenames.push_back("opencv_data\\osp_robot_3.jpg");
	filenames.push_back("opencv_data\\osp_robot_4.jpg");
	filenames.push_back("opencv_data\\osp_rc_car_1.jpg");
	filenames.push_back("opencv_data\\osp_rc_car_2.jpg");
	filenames.push_back("opencv_data\\osp_rc_car_3.jpg");
#elif 1
	filenames.push_back("opencv_data\\beaver_target.png");
	filenames.push_back("opencv_data\\melon_target.png");
	filenames.push_back("opencv_data\\puzzle.png");
	filenames.push_back("opencv_data\\lena_rgb.bmp");
#elif 0
	filenames.push_back("opencv_data\\hand_01.jpg");
	filenames.push_back("opencv_data\\hand_02.jpg");
	filenames.push_back("opencv_data\\hand_03.jpg");
	filenames.push_back("opencv_data\\hand_04.jpg");
	filenames.push_back("opencv_data\\hand_05.jpg");
	filenames.push_back("opencv_data\\hand_06.jpg");
	filenames.push_back("opencv_data\\hand_07.jpg");  // error occurred !!!
	filenames.push_back("opencv_data\\hand_08.jpg");
	filenames.push_back("opencv_data\\hand_09.jpg");
	filenames.push_back("opencv_data\\hand_10.jpg");
	filenames.push_back("opencv_data\\hand_11.jpg");
	filenames.push_back("opencv_data\\hand_12.jpg");
	filenames.push_back("opencv_data\\hand_13.jpg");
	filenames.push_back("opencv_data\\hand_14.jpg");
	filenames.push_back("opencv_data\\hand_15.jpg");
	filenames.push_back("opencv_data\\hand_16.jpg");
	filenames.push_back("opencv_data\\hand_17.jpg");
	filenames.push_back("opencv_data\\hand_18.jpg");
	filenames.push_back("opencv_data\\hand_19.jpg");
	filenames.push_back("opencv_data\\hand_20.jpg");
	filenames.push_back("opencv_data\\hand_21.jpg");
	filenames.push_back("opencv_data\\hand_22.jpg");
	filenames.push_back("opencv_data\\hand_23.jpg");
	filenames.push_back("opencv_data\\hand_24.jpg");
	filenames.push_back("opencv_data\\hand_25.jpg");
	filenames.push_back("opencv_data\\hand_26.jpg");
	filenames.push_back("opencv_data\\hand_27.jpg");
	filenames.push_back("opencv_data\\hand_28.jpg");
	filenames.push_back("opencv_data\\hand_29.jpg");
	filenames.push_back("opencv_data\\hand_30.jpg");
	filenames.push_back("opencv_data\\hand_31.jpg");
	filenames.push_back("opencv_data\\hand_32.jpg");
	filenames.push_back("opencv_data\\hand_33.jpg");
	filenames.push_back("opencv_data\\hand_34.jpg");
	filenames.push_back("opencv_data\\hand_35.jpg");
	filenames.push_back("opencv_data\\hand_36.jpg");
#elif 0
	filenames.push_back("opencv_data\\simple_hand_01.jpg");
	filenames.push_back("opencv_data\\simple_hand_02.jpg");
	filenames.push_back("opencv_data\\simple_hand_03.jpg");
	filenames.push_back("opencv_data\\simple_hand_04.jpg");
	filenames.push_back("opencv_data\\simple_hand_05.jpg");
	filenames.push_back("opencv_data\\simple_hand_06.jpg");
	filenames.push_back("opencv_data\\simple_hand_07.jpg");
	filenames.push_back("opencv_data\\simple_hand_08.jpg");
	filenames.push_back("opencv_data\\simple_hand_09.jpg");
	filenames.push_back("opencv_data\\simple_hand_10.jpg");
	filenames.push_back("opencv_data\\simple_hand_11.jpg");
	filenames.push_back("opencv_data\\simple_hand_12.jpg");
	filenames.push_back("opencv_data\\simple_hand_13.jpg");
#endif

	const char *windowName = "feature extraction";
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		IplImage *srcImage = cvLoadImage(it->c_str());
		if (NULL == srcImage)
		{
			std::cout << "fail to load image file" << std::endl;
			return;
		}

		IplImage *grayImage = NULL;
		if (1 == srcImage->nChannels)
			cvCopy(srcImage, grayImage, NULL);
		else
		{
			grayImage = cvCreateImage(cvGetSize(srcImage), srcImage->depth, 1);
			if (_stricmp(srcImage->channelSeq, "RGB") == 0)
				cvCvtColor(srcImage, grayImage, CV_RGB2GRAY);
			else if (_stricmp(srcImage->channelSeq, "BGR") == 0)
				cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);
			else
				assert(false);
			grayImage->origin = srcImage->origin;
		}

		//
		//harris_corner(srcImage, grayImage);
		//strong_corner(srcImage, grayImage);
		//surf(srcImage, grayImage);
		//star_keypoint(srcImage, grayImage);
		mser(srcImage, grayImage);

		//
		cvShowImage(windowName, srcImage);
		cvWaitKey();

		//
		cvReleaseImage(&grayImage);
		cvReleaseImage(&srcImage);
	}

	cvDestroyWindow(windowName);
}
