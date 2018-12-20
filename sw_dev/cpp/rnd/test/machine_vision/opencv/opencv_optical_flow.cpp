//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cv.h>
//#include <opencv2/legacy/legacy.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <stdexcept>


#if defined(min)
#undef min
#endif

#define __USE_ROI 1


namespace my_opencv {

// [ref] ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp.
void draw_histogram_1D(const cv::MatND &hist, const int binCount, const double maxVal, const int binWidth, const int maxHeight, cv::Mat &histImg);
void normalize_histogram(cv::MatND &hist, const double factor);

}  // namespace my_opencv

namespace {
namespace local {

//const std::string img1_filename = "../data/machine_vision/opencv/synthesized_training_image1.bmp";
//const std::string img2_filename = "../data/machine_vision/opencv/synthesized_training_image2.bmp";
//const std::string img2_filename = "../data/machine_vision/opencv/synthesized_training_image3.bmp";
//const std::string img1_filename = "../data/machine_vision/opencv/synthesized_testing_image1.bmp";
//const std::string img2_filename = "../data/machine_vision/opencv/synthesized_testing_image2.bmp";
//const std::string img2_filename = "../data/machine_vision/opencv/synthesized_testing_image3.bmp";
const std::string img1_filename = "../data/machine_vision/opencv/sample3_01.jpg";
const std::string img2_filename = "../data/machine_vision/opencv/sample3_02.jpg";

void draw_orientation_histogram(const cv::Mat &flow, const std::string &windowName, const double normalization_factor)
{
	const bool does_apply_magnitude_filtering = true;
	const double magnitude_filtering_threshold_ratio = 0.3;

	//
	std::vector<cv::Mat> flows;
	cv::split(flow, flows);

	cv::Mat flow_phase;
	cv::phase(flows[0], flows[1], flow_phase, true);  // return type: CV_32F.

	// filter by magnitude.
	if (does_apply_magnitude_filtering)
	{
		cv::Mat flow_mag;
		cv::magnitude(flows[0], flows[1], flow_mag);  // return type: CV_32F.
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(flow_mag, &minVal, &maxVal, NULL, NULL);
		const double mag_threshold = minVal + (maxVal - minVal) * magnitude_filtering_threshold_ratio;

		// TODO [check] >> magic numver, 1000 is correct ?
		flow_phase.setTo(cv::Scalar::all(1000), flow_mag < mag_threshold);
	}

	// histograms' parameters.
	const int dims = 1;
	const int bins = 360;
	const int histSize[] = { bins };
	// angle varies from 0 to 359.
	const float ranges1[] = { 0, bins };
	const float *ranges[] = { ranges1 };
	// we compute the histogram from the 0-th channel.
	const int channels[] = { 0 };
	const int bin_width = 1, bin_max_height = 100;

	// calculate histogram.
	cv::MatND hist;
	cv::calcHist(&flow_phase, 1, channels, cv::Mat(), hist, dims, histSize, ranges, true, false);

	// normalize histogram.
	my_opencv::normalize_histogram(hist, normalization_factor);

	// draw 1-D histogram.
#if 0
	double maxVal = 0.0;
	cv::minMaxLoc(hist, NULL, &maxVal, NULL, NULL);
#else
	const double maxVal = normalization_factor * 0.05;
#endif

	cv::Mat histImg(cv::Mat::zeros(bin_max_height, bins*bin_width, CV_8UC3));
	my_opencv::draw_histogram_1D(hist, bins, maxVal, bin_width, bin_max_height, histImg);

	cv::imshow(windowName, histImg);
}

void block_matching_optical_flow_algorithm()
{
#if 0
#if 0
	const int camId = -1;
	//CvCapture *capture = cvCaptureFromCAM(camId);
	CvCapture *capture = cvCreateCameraCapture(camId);
#else
	const std::string avi_filename("../data/machine_vision/opencv/tree.avi");
	//CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
	CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());
#endif

	if (!capture)
	{
		std::cerr << "Could not initialize capturing..." << std::endl;
		return;
	}

	cvNamedWindow("optical flow by block matching", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("optical flow by block matching: Horizontal Flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("optical flow by block matching: Vertical Flow", CV_WINDOW_AUTOSIZE);

	const double tol = 1.0e-10;
	const CvSize block_size = cvSize(9, 9);
	const CvSize shift_size = cvSize(5, 5);
	const CvSize max_range = cvSize(16, 16);
	const int use_previous = 0;
	IplImage *img = NULL;
	IplImage *grey = NULL, *prev_grey = NULL, *swap_temp;
	IplImage *vel_x = NULL, *vel_y = NULL;
	for (;;)
	{
		IplImage *frame = NULL;

		frame = cvQueryFrame(capture);
		if (!frame)
			break;

		if (!img)
		{
			const CvSize sz = cvGetSize(frame);
			img = cvCreateImage(sz, IPL_DEPTH_8U, 3);
			img->origin = frame->origin;
			grey = cvCreateImage(sz, IPL_DEPTH_8U, 1);
			prev_grey = cvCreateImage(sz, IPL_DEPTH_8U, 1);
			vel_x = cvCreateImage(cvSize((sz.width - block_size.width) / shift_size.width, (sz.height - block_size.height) / shift_size.height), IPL_DEPTH_32F, 1);
			vel_y = cvCreateImage(cvSize((sz.width - block_size.width) / shift_size.width, (sz.height - block_size.height) / shift_size.height), IPL_DEPTH_32F, 1);
		}

		cvCopy(frame, img, 0);
		cvCvtColor(img, grey, CV_BGR2GRAY);

		cvCalcOpticalFlowBM(prev_grey, grey, use_previous, vel_x, vel_y, lambda, term_criteria);

		for (int i = 0; i < vel_x->height; ++i)
		{
			for (int j = 0; j < vel_x->width; ++j)
			{
				const int dx = (int)cvGetReal2D(vel_x, i, j);
				const int dy = (int)cvGetReal2D(vel_y, i, j);
				if (dx*dx + dy*dy > tol)
					cvLine(img, cvPoint(j*shift_size.width, i*shift_size.height), cvPoint(j*shift_size.width+dx, i*shift_size.height+dy), CV_RGB(255,0,0), 1, 8, 0);
			}
		}

		cvShowImage("optical flow by block matching", img);
		cvShowImage("optical flow by block matching: Horizontal Flow", vel_x);
		cvShowImage("optical flow by block matching: Vertical Flow", vel_y);

		cvWaitKey(10);

		CV_SWAP(prev_grey, grey, swap_temp);
	}

	cvDestroyWindow("optical flow by block matching");
	cvDestroyWindow("optical flow by block matching: Horizontal Flow");
	cvDestroyWindow("optical flow by block matching: Vertical Flow");

	cvReleaseImage(&img);  img = NULL;
	cvReleaseImage(&prev_grey);  prev_grey = NULL;
	cvReleaseImage(&grey);  grey = NULL;
	cvReleaseImage(&vel_x);  vel_x = NULL;
	cvReleaseImage(&vel_y);  vel_y = NULL;

	cvReleaseCapture(&capture);
#else
	IplImage *img1 = cvLoadImage(img1_filename.c_str(), CV_LOAD_IMAGE_COLOR);
	IplImage *img2 = cvLoadImage(img2_filename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (!img1 || !img2) return;

	IplImage *grey1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	IplImage *grey2 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 1);

	cvCvtColor(img1, grey1, CV_BGR2GRAY);
	cvCvtColor(img2, grey2, CV_BGR2GRAY);

	const CvSize block_size = cvSize(9, 9);
	const CvSize shift_size = cvSize(5, 5);
	const CvSize max_range = cvSize(16, 16);
	const int use_previous = 0;
	IplImage *vel_x = cvCreateImage(cvSize((grey1->width - block_size.width) / shift_size.width, (grey1->height - block_size.height) / shift_size.height), IPL_DEPTH_32F, 1);
	IplImage *vel_y = cvCreateImage(cvSize((grey1->width - block_size.width) / shift_size.width, (grey1->height - block_size.height) / shift_size.height), IPL_DEPTH_32F, 1);
	cvCalcOpticalFlowBM(grey2, grey1, block_size, shift_size, max_range, use_previous, vel_x, vel_y);

	//
	const double tol = 1.0e-10;
	for (int i = 0; i < vel_x->height; ++i)
	{
		for (int j = 0; j < vel_x->width; ++j)
		{
			const int dx = (int)cvGetReal2D(vel_x, i, j);
			const int dy = (int)cvGetReal2D(vel_y, i, j);
			if (dx*dx + dy*dy > tol)
				cvLine(img1, cvPoint(j*shift_size.width, i*shift_size.height), cvPoint(j*shift_size.width+dx, i*shift_size.height+dy), CV_RGB(255,0,0), 1, 8, 0);
		}
	}

	cvSaveImage((img1_filename + "_bm_vel_x.bmp").c_str(), vel_x);
	cvSaveImage((img1_filename + "_bm_vel_y.bmp").c_str(), vel_y);

	cvNamedWindow("optical flow by block matching", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("optical flow by block matching: Horizontal Flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("optical flow by block matching: Vertical Flow", CV_WINDOW_AUTOSIZE);

	cvShowImage("optical flow by block matching", img1);
	cvShowImage("optical flow by block matching: Horizontal Flow", vel_x);
	cvShowImage("optical flow by block matching: Vertical Flow", vel_y);

	cvWaitKey(0);

	cvDestroyWindow("optical flow by block matching");
	cvDestroyWindow("optical flow by block matching: Horizontal Flow");
	cvDestroyWindow("optical flow by block matching: Vertical Flow");

	cvReleaseImage(&img1);  img1 = NULL;
	cvReleaseImage(&img2);  img2 = NULL;
	cvReleaseImage(&grey1);  grey1 = NULL;
	cvReleaseImage(&grey2);  grey2 = NULL;
	cvReleaseImage(&vel_x);  vel_x = NULL;
	cvReleaseImage(&vel_y);  vel_y = NULL;
#endif
}

void Horn_Schunck_optical_flow_algorithm()
{
#if 0
#if 0
	const int camId = -1;
	//CvCapture *capture = cvCaptureFromCAM(camId);
	CvCapture *capture = cvCreateCameraCapture(camId);
#else
	const std::string avi_filename("../data/machine_vision/opencv/tree.avi");
	//CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
	CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());
#endif

	if (!capture)
	{
		std::cerr << "Could not initialize capturing..." << std::endl;
		return;
	}

	cvNamedWindow("Horn Schunck optical flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Horn Schunck optical flow: Horizontal Flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Horn Schunck optical flow: Vertical Flow", CV_WINDOW_AUTOSIZE);

	const double tol = 1.0e-10;
	const int use_previous = 0;
	const double lambda = 0.5;
	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	IplImage *img = NULL;
	IplImage *grey = NULL, *prev_grey = NULL, *swap_temp;
	IplImage *vel_x = NULL, *vel_y = NULL;
	for (;;)
	{
		IplImage *frame = NULL;

		frame = cvQueryFrame(capture);
		if (!frame)
			break;

		if (!img)
		{
			const CvSize sz = cvGetSize(frame);
			img = cvCreateImage(sz, IPL_DEPTH_8U, 3);
			img->origin = frame->origin;
			grey = cvCreateImage(sz, IPL_DEPTH_8U, 1);
			prev_grey = cvCreateImage(sz, IPL_DEPTH_8U, 1);
			vel_x = cvCreateImage(sz, IPL_DEPTH_32F, 1);
			vel_y = cvCreateImage(sz, IPL_DEPTH_32F, 1);
		}

		cvCopy(frame, img, 0);
		cvCvtColor(img, grey, CV_BGR2GRAY);

		cvCalcOpticalFlowHS(prev_grey, grey, use_previous, vel_x, vel_y, lambda, term_criteria);

		for (int i = 0; i < img->height; ++i)
		{
			for (int j = 0; j < img->width; ++j)
			{
				const int dx = (int)cvGetReal2D(vel_x, i, j);
				const int dy = (int)cvGetReal2D(vel_y, i, j);
				if (dx*dx + dy*dy > tol)
					cvLine(img, cvPoint(j, i), cvPoint(j+dx, i+dy), CV_RGB(255,0,0), 1, 8, 0);
			}
		}

		cvShowImage("Horn Schunck optical flow", img);
		cvShowImage("Horn Schunck optical flow: Horizontal Flow", vel_x);
		cvShowImage("Horn Schunck optical flow: Vertical Flow", vel_y);

		cvWaitKey(10);

		CV_SWAP(prev_grey, grey, swap_temp);
	}

	cvDestroyWindow("Horn Schunck optical flow");
	cvDestroyWindow("Horn Schunck optical flow: Horizontal Flow");
	cvDestroyWindow("Horn Schunck optical flow: Vertical Flow");

	cvReleaseImage(&img);  img = NULL;
	cvReleaseImage(&prev_grey);  prev_grey = NULL;
	cvReleaseImage(&grey);  grey = NULL;
	cvReleaseImage(&vel_x);  vel_x = NULL;
	cvReleaseImage(&vel_y);  vel_y = NULL;

	cvReleaseCapture(&capture);
#else
	IplImage *img1 = cvLoadImage(img1_filename.c_str(), CV_LOAD_IMAGE_COLOR);
	IplImage *img2 = cvLoadImage(img2_filename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (!img1 || !img2) return;

	IplImage *grey1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	IplImage *grey2 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 1);
	IplImage *vel_x = cvCreateImage(cvGetSize(img1), IPL_DEPTH_32F, 1);
	IplImage *vel_y = cvCreateImage(cvGetSize(img1), IPL_DEPTH_32F, 1);

	cvCvtColor(img1, grey1, CV_BGR2GRAY);
	cvCvtColor(img2, grey2, CV_BGR2GRAY);

	const int use_previous = 0;
	const double lambda = 0.5;
	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	cvCalcOpticalFlowHS(grey1, grey2, use_previous, vel_x, vel_y, lambda, term_criteria);

	//
	const double tol = 1.0e-10;
	for (int i = 0; i < img1->height; ++i)
	{
		for (int j = 0; j < img1->width; ++j)
		{
			const int dx = (int)cvGetReal2D(vel_x, i, j);
			const int dy = (int)cvGetReal2D(vel_y, i, j);
			if (dx*dx + dy*dy > tol)
				cvLine(img1, cvPoint(j, i), cvPoint(j+dx, i+dy), CV_RGB(255,0,0), 1, 8, 0);
		}
	}

	cvSaveImage((img1_filename + "_hs_vel_x.bmp").c_str(), vel_x);
	cvSaveImage((img1_filename + "_hs_vel_y.bmp").c_str(), vel_y);

	cvNamedWindow("Horn Schunck optical flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Horn Schunck optical flow: Horizontal Flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Horn Schunck optical flow: Vertical Flow", CV_WINDOW_AUTOSIZE);

	cvShowImage("Horn Schunck optical flow", img1);
	cvShowImage("Horn Schunck optical flow: Horizontal Flow", vel_x);
	cvShowImage("Horn Schunck optical flow: Vertical Flow", vel_y);

	cvWaitKey(0);

	cvDestroyWindow("Horn Schunck optical flow");
	cvDestroyWindow("Horn Schunck optical flow: Horizontal Flow");
	cvDestroyWindow("Horn Schunck optical flow: Vertical Flow");

	cvReleaseImage(&img1);  img1 = NULL;
	cvReleaseImage(&img2);  img2 = NULL;
	cvReleaseImage(&grey1);  grey1 = NULL;
	cvReleaseImage(&grey2);  grey2 = NULL;
	cvReleaseImage(&vel_x);  vel_x = NULL;
	cvReleaseImage(&vel_y);  vel_y = NULL;
#endif
}

void Lucas_Kanade_optical_flow_algorithm()
{
#if 0
#if 0
	const int camId = -1;
	//CvCapture *capture = cvCaptureFromCAM(camId);
	CvCapture *capture = cvCreateCameraCapture(camId);
#else
	const std::string avi_filename("../data/machine_vision/opencv/tree.avi");
	//CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
	CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());
#endif

	if (!capture)
	{
		std::cerr << "Could not initialize capturing..." << std::endl;
		return;
	}

	cvNamedWindow("Lucas Kanade optical flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Lucas Kanade optical flow: Horizontal Flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Lucas Kanade optical flow: Vertical Flow", CV_WINDOW_AUTOSIZE);

	const int win_size = 5;  // odd number.
	IplImage *img = NULL;
	IplImage *grey = NULL, *prev_grey = NULL, *swap_temp;
	IplImage *vel_x = NULL, *vel_y = NULL;
	const double tol = 1.0e-10;
	for (;;)
	{
		IplImage *frame = NULL;

		frame = cvQueryFrame(capture);
		if (!frame)
			break;

		if (!img)
		{
			const CvSize sz = cvGetSize(frame);
			img = cvCreateImage(sz, IPL_DEPTH_8U, 3);
			img->origin = frame->origin;
			grey = cvCreateImage(sz, IPL_DEPTH_8U, 1);
			prev_grey = cvCreateImage(sz, IPL_DEPTH_8U, 1);
			vel_x = cvCreateImage(sz, IPL_DEPTH_32F, 1);
			vel_y = cvCreateImage(sz, IPL_DEPTH_32F, 1);
		}

		cvCopy(frame, img, 0);
		cvCvtColor(img, grey, CV_BGR2GRAY);

		cvCalcOpticalFlowLK(prev_grey, grey, cvSize(win_size, win_size), vel_x, vel_y);

		for (int i = 0; i < img->height; ++i)
		{
			for (int j = 0; j < img->width; ++j)
			{
				const int dx = (int)cvGetReal2D(vel_x, i, j);
				const int dy = (int)cvGetReal2D(vel_y, i, j);
				if (dx*dx + dy*dy > tol)
					cvLine(img, cvPoint(j, i), cvPoint(j+dx, i+dy), CV_RGB(255,0,0), 1, 8, 0);
			}
		}

		cvShowImage("Lucas Kanade optical flow", img);
		cvShowImage("Lucas Kanade optical flow: Horizontal Flow", vel_x);
		cvShowImage("Lucas Kanade optical flow: Vertical Flow", vel_y);

		cvWaitKey(10);

		CV_SWAP(prev_grey, grey, swap_temp);
	}

	cvDestroyWindow("Lucas Kanade optical flow");
	cvDestroyWindow("Lucas Kanade optical flow: Horizontal Flow");
	cvDestroyWindow("Lucas Kanade optical flow: Vertical Flow");

	cvReleaseImage(&img);  img = NULL;
	cvReleaseImage(&prev_grey);  prev_grey = NULL;
	cvReleaseImage(&grey);  grey = NULL;
	cvReleaseImage(&vel_x);  vel_x = NULL;
	cvReleaseImage(&vel_y);  vel_y = NULL;

	cvReleaseCapture(&capture);
#else
	IplImage *img1 = cvLoadImage(img1_filename.c_str(), CV_LOAD_IMAGE_COLOR);
	IplImage *img2 = cvLoadImage(img2_filename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (!img1 || !img2) return;

	IplImage *grey1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	IplImage *grey2 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 1);
	IplImage *vel_x = cvCreateImage(cvGetSize(img1), IPL_DEPTH_32F, 1);
	IplImage *vel_y = cvCreateImage(cvGetSize(img1), IPL_DEPTH_32F, 1);

	cvCvtColor(img1, grey1, CV_BGR2GRAY);
	cvCvtColor(img2, grey2, CV_BGR2GRAY);

	const int win_size = 5;  // odd number.
	cvCalcOpticalFlowLK(grey1, grey2, cvSize(win_size, win_size), vel_x, vel_y);

	//
	const double tol = 1.0e-10;
	for (int i = 0; i < img1->height; ++i)
	{
		for (int j = 0; j < img1->width; ++j)
		{
			const int dx = (int)cvGetReal2D(vel_x, i, j);
			const int dy = (int)cvGetReal2D(vel_y, i, j);
			if (dx*dx + dy*dy > tol)
				cvLine(img1, cvPoint(j, i), cvPoint(j+dx, i+dy), CV_RGB(255,0,0), 1, 8, 0);
		}
	}

	cvSaveImage((img1_filename + "_lk_vel_x.bmp").c_str(), vel_x);
	cvSaveImage((img1_filename + "_lk_vel_y.bmp").c_str(), vel_y);

	cvNamedWindow("Lucas Kanade optical flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Lucas Kanade optical flow: Horizontal Flow", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Lucas Kanade optical flow: Vertical Flow", CV_WINDOW_AUTOSIZE);

	cvShowImage("Lucas Kanade optical flow", img1);
	cvShowImage("Lucas Kanade optical flow: Horizontal Flow", vel_x);
	cvShowImage("Lucas Kanade optical flow: Vertical Flow", vel_y);

	cvWaitKey(0);

	cvDestroyWindow("Lucas Kanade optical flow");
	cvDestroyWindow("Lucas Kanade optical flow: Horizontal Flow");
	cvDestroyWindow("Lucas Kanade optical flow: Vertical Flow");

	cvReleaseImage(&img1);  img1 = NULL;
	cvReleaseImage(&img2);  img2 = NULL;
	cvReleaseImage(&grey1);  grey1 = NULL;
	cvReleaseImage(&grey2);  grey2 = NULL;
	cvReleaseImage(&vel_x);  vel_x = NULL;
	cvReleaseImage(&vel_y);  vel_y = NULL;
#endif
}

IplImage *image = 0;
int add_remove_pt = 0;
CvPoint pt;

void on_mouse(int evt, int x, int y, int flags, void *param)
{
    if (!image)
        return;

    if (image->origin)
        y = image->height - y;

    if (CV_EVENT_LBUTTONDOWN == evt)
    {
        pt = cvPoint(x, y);
        add_remove_pt = 1;
    }
}

void pyramid_Lucas_Kanade_optical_flow_algorithm_1()
{
#if 0
	IplImage *grey = 0, *prev_grey = 0, *pyramid = 0, *prev_pyramid = 0, *swap_temp;

	int win_size = 10;
	const int MAX_COUNT = 500;
	CvPoint2D32f *points[2] = {0, 0}, *swap_points;
	char *status = 0;
	int count = 0;
	int need_to_init = 1;
	int night_mode = 0;
	int flags = 0;

#if 0
	const int camId = -1;
	//CvCapture *capture = cvCaptureFromCAM(camId);
	CvCapture *capture = cvCreateCameraCapture(camId);
#else
	const std::string avi_filename("../data/machine_vision/opencv/tree.avi");
	//CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
	CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());
#endif

	if (!capture)
	{
		std::cerr << "Could not initialize capturing..." << std::endl;
		return;
	}

	// print a welcome message, and the OpenCV version.
	printf(
		"Welcome to pyramid Lucas-Kanade, using OpenCV version %s (%d.%d.%d)\n",
		CV_VERSION,
		CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION
	);

	printf(
		"Hot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n"
	);

	cvNamedWindow("pyramid Lucas Kanade optical flow", CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback("pyramid Lucas Kanade optical flow", on_mouse, 0);

	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	for (;;)
	{
		IplImage *frame = 0;
		int i, k, c;

		frame = cvQueryFrame(capture);
		if (!frame)
			break;

		if (!image)
		{
			// allocate all the buffers.
			image = cvCreateImage(cvGetSize(frame), 8, 3);
			image->origin = frame->origin;
			grey = cvCreateImage(cvGetSize(frame), 8, 1);
			prev_grey = cvCreateImage(cvGetSize(frame), 8, 1);
			pyramid = cvCreateImage(cvGetSize(frame), 8, 1);
			prev_pyramid = cvCreateImage(cvGetSize(frame), 8, 1);
			points[0] = (CvPoint2D32f*)cvAlloc(MAX_COUNT * sizeof(points[0][0]));
			points[1] = (CvPoint2D32f*)cvAlloc(MAX_COUNT * sizeof(points[0][0]));
			status = (char*)cvAlloc(MAX_COUNT);
			flags = 0;
		}

		cvCopy(frame, image, 0);
		cvCvtColor(image, grey, CV_BGR2GRAY);

		if (night_mode)
			cvZero(image);

		if (need_to_init)
		{
			// automatic initialization.
			IplImage *eig = cvCreateImage(cvGetSize(grey), 32, 1);
			IplImage *temp = cvCreateImage(cvGetSize(grey), 32, 1);
			double quality = 0.01;
			double min_distance = 10;

			count = MAX_COUNT;
			cvGoodFeaturesToTrack(
				grey, eig, temp, points[1], &count,
				quality, min_distance, 0, 3, 0, 0.04
			);
			cvFindCornerSubPix(
				grey, points[1], count,
				cvSize(win_size, win_size), cvSize(-1,-1),
				term_criteria
			);
			cvReleaseImage(&eig);
			cvReleaseImage(&temp);

			add_remove_pt = 0;
		}
		else if (count > 0)
		{
			cvCalcOpticalFlowPyrLK(
				prev_grey, grey, prev_pyramid, pyramid,
				points[0], points[1], count, cvSize(win_size, win_size), 3, status, 0,
				term_criteria, flags
			);
			flags |= CV_LKFLOW_PYR_A_READY;
			for (i = k = 0; i < count; ++i)
			{
				if (add_remove_pt)
				{
					double dx = pt.x - points[1][i].x;
					double dy = pt.y - points[1][i].y;

					if (dx*dx + dy*dy <= 25)
					{
						add_remove_pt = 0;
						continue;
					}
				}

				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				cvCircle(image, cvPointFrom32f(points[1][i]), 3, CV_RGB(0, 255, 0), CV_FILLED, cv::LINE_8, 0);
			}
			count = k;
		}

		if (add_remove_pt && count < MAX_COUNT)
		{
			points[1][count++] = cvPointTo32f(pt);
			cvFindCornerSubPix(
				grey, points[1] + count - 1, 1,
				cvSize(win_size, win_size), cvSize(-1, -1),
				term_criteria
			);
			add_remove_pt = 0;
		}

		CV_SWAP(prev_grey, grey, swap_temp);
		CV_SWAP(prev_pyramid, pyramid, swap_temp);
		CV_SWAP(points[0], points[1], swap_points);
		need_to_init = 0;
		cvShowImage("pyramid Lucas Kanade optical flow", image);

		c = cvWaitKey(10);
		if (27 == (char)c)
			break;
		switch ((char)c)
		{
		case 'r':
			need_to_init = 1;
			break;
		case 'c':
			count = 0;
			break;
		case 'n':
			night_mode ^= 1;
			break;
		default:
			break;
		}
	}

	cvDestroyWindow("pyramid Lucas Kanade optical flow");

	cvReleaseImage(&image);  image = NULL;
	cvReleaseImage(&grey);  grey = NULL;
	cvReleaseImage(&prev_grey);  prev_grey = NULL;
	cvReleaseImage(&pyramid);  pyramid = NULL;
	cvReleaseImage(&prev_pyramid);  prev_pyramid = NULL;
	//cvReleaseImage(&swap_temp);  swap_temp = NULL;
	cvFree(&(points[0]));
	cvFree(&(points[1]));
	cvFree(&status);

	cvReleaseCapture(&capture);
#else
	IplImage *img1 = cvLoadImage(img1_filename.c_str(), CV_LOAD_IMAGE_COLOR);
	IplImage *img2 = cvLoadImage(img2_filename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (!img1 || !img2) return;

	IplImage *grey1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	IplImage *grey2 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 1);
	IplImage *vel_x = cvCreateImage(cvGetSize(img1), IPL_DEPTH_32F, 1);
	IplImage *vel_y = cvCreateImage(cvGetSize(img1), IPL_DEPTH_32F, 1);

	cvCvtColor(img1, grey1, CV_BGR2GRAY);
	cvCvtColor(img2, grey2, CV_BGR2GRAY);

	const int win_size = 10;  // odd number ???
	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);

#if defined(__USE_ROI)
	const CvSize imgSize1 = cvGetSize(grey1);
	const CvSize imgSize2 = cvGetSize(grey2);
	const CvRect roi = cvRect(imgSize1.width / 8, imgSize1.height / 2, std::min(imgSize1.width, imgSize2.width) / 2, std::min(imgSize1.height, imgSize2.height) / 4);
	cvSetImageROI(grey1, roi);
	cvSetImageROI(grey2, roi);
#endif

#if 0
	const int count = 1;
	const int pt_x = 90, pt_y = 103;

    CvPoint2D32f *features1 = NULL, *features2 = NULL;
	char *status = (char*)cvAlloc(1);
	*status = 0;

	features1 = (CvPoint2D32f*)cvAlloc(sizeof(CvPoint2D32f));
	features2 = (CvPoint2D32f*)cvAlloc(sizeof(CvPoint2D32f));
    features1->x = (float)pt_x;
    features1->y = (float)pt_y;
    *features2 = *features1;
#else
	const int MAX_COUNT = 500;
	int count = 0;

	CvPoint2D32f *features1 = (CvPoint2D32f*)cvAlloc(MAX_COUNT * sizeof(CvPoint2D32f));
	CvPoint2D32f *features2 = (CvPoint2D32f*)cvAlloc(MAX_COUNT * sizeof(CvPoint2D32f));
	char *status = (char*)cvAlloc(MAX_COUNT);
	memset(status, 0 , sizeof(char) * MAX_COUNT);

	{
		IplImage *eig = cvCreateImage(cvGetSize(grey1), 32, 1);
		IplImage *temp = cvCreateImage(cvGetSize(grey1), 32, 1);
		const double quality = 0.01;
		const double min_distance = 10;

		count = MAX_COUNT;
		cvGoodFeaturesToTrack(
			grey1, eig, temp, features1, &count,
			quality, min_distance, 0, 3, 0, 0.04
		);
		cvFindCornerSubPix(
			grey1, features1, count,
			cvSize(win_size, win_size), cvSize(-1, -1),
			term_criteria
		);

		for (int f = 0; f < count; ++f)
			features2[f] = features1[f];

		cvReleaseImage(&eig);
		cvReleaseImage(&temp);
	}
#endif

	const int level = 5;
	//const int flags = CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY;
	const int flags = CV_LKFLOW_INITIAL_GUESSES;
	cvCalcOpticalFlowPyrLK(
		grey1, grey2, NULL, NULL,
		features1, features2, count, cvSize(win_size, win_size), level, status, NULL,
		term_criteria, flags
	);
	std::cout << "# of feature points: " << count << std::endl;

	{
		CvPoint ptSelected;
		const int radius = 3;
		for (int f = 0; f < count; ++f)
			if (1 == status[f])
			{
				const int r = rand() % 256, g = rand() % 256, b = rand() % 256;
#if defined(__USE_ROI)
				ptSelected.x = (int)features1[f].x + roi.x;
				ptSelected.y = (int)features1[f].y + roi.y;
				cvCircle(img1, ptSelected, radius, CV_RGB(r, g, b), CV_FILLED, 8, 0);
				ptSelected.x = (int)features2[f].x + roi.x;
				ptSelected.y = (int)features2[f].y + roi.y;
				cvCircle(img2, ptSelected, radius, CV_RGB(r, g, b), CV_FILLED, 8, 0);
#else
				ptSelected.x = (int)features1[f].x;
				ptSelected.y = (int)features1[f].y;
				cvCircle(img1, ptSelected, radius, CV_RGB(r, g, b), CV_FILLED, 8, 0);
				ptSelected.x = (int)features2[f].x;
				ptSelected.y = (int)features2[f].y;
				cvCircle(img2, ptSelected, radius, CV_RGB(r, g, b), CV_FILLED, 8, 0);
#endif
			}

		cvNamedWindow("pyramid Lucas Kanade optical flow: Image 1", CV_WINDOW_AUTOSIZE);
		cvNamedWindow("pyramid Lucas Kanade optical flow: Image 2", CV_WINDOW_AUTOSIZE);

		cvShowImage("pyramid Lucas Kanade optical flow: Image 1", img1);
		cvShowImage("pyramid Lucas Kanade optical flow: Image 2", img2);

		cvWaitKey(0);

		cvDestroyWindow("pyramid Lucas Kanade optical flow: Image 1");
		cvDestroyWindow("pyramid Lucas Kanade optical flow: Image 2");
	}

#if defined(__USE_ROI)
	cvResetImageROI(grey1);
	cvResetImageROI(grey2);
#endif

	cvReleaseImage(&img1);  img1 = NULL;
	cvReleaseImage(&img2);  img2 = NULL;
	cvReleaseImage(&grey1);  grey1 = NULL;
	cvReleaseImage(&grey2);  grey2 = NULL;
	cvReleaseImage(&vel_x);  vel_x = NULL;
	cvReleaseImage(&vel_y);  vel_y = NULL;
	cvFree(&features1);
	cvFree(&features2);
	cvFree(&status);
#endif
}

void drawOpticalFlowMap(const cv::Mat &flow, cv::Mat &cflowmap, const int step, const double scale, const cv::Scalar &color, const cv::Point &start_pt, const double mag_threshold)
{
	for (int y = 0; y < flow.rows; y += step)
		for (int x = 0; x < flow.cols; x += step)
		{
			const cv::Point2f &fxy = flow.at<cv::Point2f>(y, x);
			if (std::sqrt(fxy.x*fxy.x + fxy.y*fxy.y) < mag_threshold) continue;
			cv::line(cflowmap, start_pt + cv::Point(x, y), start_pt + cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color, 1, cv::LINE_8, 0);
			//cv::circle(cflowmap, start_pt + cv::Point(x, y), 1, color, cv::FILLED, cv::LINE_8, 0);
		}
}

void pyramid_Lucas_Kanade_optical_flow_algorithm_2()
{
	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	//const bool b1 = capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	//const bool b2 = capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	const std::string windowName1("pyramid Lucas Kanade optical flow");
	const std::string windowName2("pyramid Lucas Kanade optical flow - orientation histogram");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

    const int MAX_COUNT = 1000;
    const cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
    const cv::Size winSize(10,10);

	cv::Mat prev_gray, gray, tmp_gray, cflow, frame;
	std::vector<cv::Point2f> prev_points, curr_points;
	for (;;)
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		cv::cvtColor(frame, tmp_gray, cv::COLOR_BGR2GRAY);

#if 1
		cv::pyrDown(tmp_gray, gray, cv::Size());
#else
		cv::resize(tmp_gray, gray, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
#endif

		if (!prev_gray.empty() && !gray.empty())
		{
			if (prev_points.empty())
			{
				cv::goodFeaturesToTrack(
					prev_gray,		// the input 8-bit or floating-point 32-bit, single-channel image
					prev_points,	// the output vector of detected corners
					MAX_COUNT,		// the maximum number of corners to return
					0.001,			// characterizes the minimal accepted quality of image corners
					1,				// the minimum possible Euclidean distance between the returned corners
					cv::Mat(),		// the optional region of interest.
					3,				// size of the averaging block for computing derivative covariation matrix over each pixel neighborhood
					false,			// indicates, whether to use Harris operator or cv::cornerMinEigenVal
					0.04			// free parameter of Harris detector
				);
				cv::cornerSubPix(prev_gray, prev_points, winSize, cv::Size(-1,-1), termcrit);
			}

            std::vector<unsigned char> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(
				prev_gray, gray,	// the first & second 8-bit single-channel or 3-channel input images
				prev_points,		// vector of points for which the flow needs to be found
				curr_points,		// the output vector of points containing the calculated new positions of the input features in the second image
				status,				// the output status vector
				err,				// the output vector that will contain the difference between patches around the original and moved points
				winSize,			// size of the search window at each pyramid level
				3,					// 0-based maximal pyramid level number
				termcrit,			// the termination criteria of the iterative search algorithm
				0					// the relative weight of the spatial image derivatives impact to the optical flow estimation
			);

			//
			cv::cvtColor(prev_gray, cflow, cv::COLOR_GRAY2BGR);

			cv::Mat flow(gray.rows, gray.cols, CV_32FC2, cv::Scalar::all(0));

			size_t k = 0;
			for (std::vector<unsigned char>::const_iterator it = status.begin(); it != status.end(); ++it, ++k)
			{
				// FIXME [delete] >>
				//std::cout << k << ": (" << prev_points[k].x << "," << prev_points[k].y << "), (" << curr_points[k].x << "," << curr_points[k].y << ")" << std::endl;

				if (!*it) continue;

				const int x = prev_points[k].x < 0.0 ? 0 : (prev_points[k].x >= flow.cols ? flow.cols-1 : cvRound(prev_points[k].x));
				const int y = prev_points[k].y < 0.0 ? 0 : (prev_points[k].y >= flow.rows ? flow.rows-1 : cvRound(prev_points[k].y));
				flow.at<cv::Point2f>(y, x) = curr_points[k] - prev_points[k];

				cv::circle(cflow, curr_points[k], 3, CV_RGB(0, 255, 0), -1, 8, 0);
			}

			//
			drawOpticalFlowMap(flow, cflow, 5, 1.5, CV_RGB(255, 0, 0), cv::Point(0, 0), 1.0);
			cv::imshow(windowName1, cflow);

			draw_orientation_histogram(flow, windowName2, MAX_COUNT);
		}

		if (cv::waitKey(1) >= 0)
			break;

		std::swap(prev_points, curr_points);
		std::swap(prev_gray, gray);
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

void Farneback_motion_estimation_algorithm()
{
#if 1
	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	//const bool b1 = capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	//const bool b2 = capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	const std::string windowName1("Farneback optical flow");
	const std::string windowName2("Farneback optical flow - orientation histogram");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	cv::Mat prev_gray, gray, tmp_gray, flow, cflow, frame;
	for (;;)
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		cv::cvtColor(frame, tmp_gray, cv::COLOR_BGR2GRAY);

#if 1
		cv::pyrDown(tmp_gray, gray, cv::Size());
#else
		cv::resize(tmp_gray, gray, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
#endif

		if (prev_gray.data)
		{
			cv::calcOpticalFlowFarneback(
				prev_gray, gray,	// the first & second 8-bit single-channel input image
				flow,				// the computed flow image. type: CV_32FC2
				0.25,				// the image scale to build the pyramids
				7,					// the number of pyramid layers
				15,					// the averaging window size
				3,					// the number of iterations the algorithm does at each pyramid level
				5,					// size of the pixel neighborhood used to find polynomial expansion
				1.1,				// standard deviation of the Gaussian that is used to smooth derivatives that are used as a basis for the polynomial expansion
				0					// the operation flags
			);

			//
			cv::cvtColor(prev_gray, cflow, cv::COLOR_GRAY2BGR);
			drawOpticalFlowMap(flow, cflow, 5, 1.5, CV_RGB(0, 255, 0), cv::Point(0, 0), 1.0);
			cv::imshow(windowName1, cflow);

			draw_orientation_histogram(flow, windowName2, 1000);
		}

		if (cv::waitKey(1) >= 0)
			break;

		std::swap(prev_gray, gray);
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
#else
	//cv::Mat img1 = cv::imread(img1_filename, CV_LOAD_IMAGE_COLOR);
	//cv::Mat img2 = cv::imread(img2_filename, CV_LOAD_IMAGE_COLOR);
	//if (img1.empty() || img2.empty()) return;
	IplImage *img01 = cvLoadImage(img1_filename.c_str(), CV_LOAD_IMAGE_COLOR);
	IplImage *img02 = cvLoadImage(img2_filename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (!img01 || !img02) return;
	cv::Mat img1(img01, false);
	cv::Mat img2(img02, false);
	if (img1.empty() || img2.empty()) return;

	cv::Mat grey1, grey2;
	cv::cvtColor(img1, grey1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img2, grey2, cv::COLOR_BGR2GRAY);

	//const cv::Rect roi(0, 0, img1.cols / 2, img1.rows / 2);
	const cv::Rect roi(img1.cols / 2, img1.rows / 2, img1.cols / 2, img1.rows / 2);

	//const cv::Mat grey1_roi(grey1, roi), grey2_roi(grey2, roi);
	const cv::Mat &grey1_roi = grey1(roi), &grey2_roi = grey2(roi);

	cv::Mat flow;
	cv::calcOpticalFlowFarneback(grey1_roi, grey2_roi, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	cv::Mat cflow;
	cv::cvtColor(grey1, cflow, cv::COLOR_GRAY2BGR);
	drawOpticalFlowMap(flow, cflow, 5, 1.5, CV_RGB(0, 255, 0), cv::Point(roi.x, roi.y), 1.0);

	const std::string windowName("Farneback optical flow");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName, cflow);

	cv::waitKey(0);

	cv::destroyWindow(windowName);
#endif
}

// REF [file] >> ${OPENCV_HOME}/samples/cpp/simpleflow_demo.cpp.
void motion_estimation_algorithm_by_simpleflow()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void optical_flow()
{
	//local::block_matching_optical_flow_algorithm();
	//local::Horn_Schunck_optical_flow_algorithm();
	//local::Lucas_Kanade_optical_flow_algorithm();
	//local::pyramid_Lucas_Kanade_optical_flow_algorithm_1();
	//local::pyramid_Lucas_Kanade_optical_flow_algorithm_2();
	local::Farneback_motion_estimation_algorithm();
	//local::motion_estimation_algorithm_by_simpleflow();  // Not yet implemented.
}

}  // namespace my_opencv
