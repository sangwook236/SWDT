//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/background_segm.hpp>
#include <string>
#include <iostream>


namespace {
namespace local {

#if 0
void background_segmentation()
{
#if 1
	const int camId = -1;
	//CvCapture *capture = cvCaptureFromCAM(camId);
	CvCapture *capture = cvCreateCameraCapture(camId);
#else
	const std::string avi_filename("machine_vision_data\\opencv\\tree.avi");
	//CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
	CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());
#endif

    if (!capture)
    {
		std::cout << "can not open camera or video file" << std::endl;
        return;
    }

	cvNamedWindow("BG", 1);
	cvNamedWindow("FG", 1);

	CvBGStatModel *bg_model = NULL;

    bool update_bg_model = true;
	IplImage *frame = NULL, *image = NULL;
	for (int fr = 1; ; ++fr)
	{
		frame = cvQueryFrame(capture);
		if (!frame)
		{
			std::cout << "can not read data from the video source" << std::endl;
			continue;
		}

		if (NULL == image) image = cvCloneImage(frame);
		else cvCopy(frame, image);

		// create BG model
		if (NULL == bg_model && image)
		{
			bg_model = cvCreateGaussianBGModel(image);
			//bg_model = cvCreateFGDStatModel(image);
		}

		double t = (double)cvGetTickCount();

		cvUpdateBGStatModel(image, bg_model, update_bg_model ? -1 : 0);

		std::cout << "frame: " << fr << ", time: " << ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000.) << std::endl;

		cvShowImage("BG", bg_model->background);
		cvShowImage("FG", bg_model->foreground);

		const char k = cvWaitKey(1);
		if (27 == k) break;

		if (' ' == k)
		{
			update_bg_model = !update_bg_model;
			std::cout << "background update is " << (update_bg_model ? "on" : "off") << std::endl;
		}
	}

	cvReleaseBGStatModel(&bg_model);

    cvDestroyWindow("BG");
    cvDestroyWindow("FG");
	cvReleaseCapture(&capture);
}

CvBGCodeBookModel *model = 0;
const int NCHANNELS = 3;
bool ch[NCHANNELS] = { true, true, true }; // This sets what channels should be adjusted for background bounds

void change_detection_using_codebook()
{
	const char *filename = 0;
	IplImage *rawImage = 0, *yuvImage = 0; //yuvImage is for codebook method
	IplImage *ImaskCodeBook = 0, *ImaskCodeBookCC = 0;

	int c, n, nframes = 0;
	int nframesToLearnBG = 20;

	model = cvCreateBGCodeBookModel();

	//Set color thresholds to default values
	model->modMin[0] = 3;
	model->modMin[1] = model->modMin[2] = 3;
	model->modMax[0] = 10;
	model->modMax[1] = model->modMax[2] = 10;
	model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 10;

	bool pause = false;
	bool singlestep = false;

#if 1
	const int camId = -1;
	//CvCapture *capture = cvCaptureFromCAM(camId);
	CvCapture *capture = cvCreateCameraCapture(camId);
#else
	const std::string avi_filename("machine_vision_data\\opencv\\tree.avi");
	//CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
	CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());
#endif

	if (!capture)
	{
		std::cout << "Can not initialize video capturing\n\n";
		//help();
		return;
	}

	// MAIN PROCESSING LOOP:
	for (;;)
	{
		if (!pause)
		{
			rawImage = cvQueryFrame(capture);
			++nframes;
			if (!rawImage)
				break;
		}
		if (singlestep)
			pause = true;

		// First time:
		if (nframes == 1 && rawImage)
		{
			// CODEBOOK METHOD ALLOCATION
			yuvImage = cvCloneImage(rawImage);
			ImaskCodeBook = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
			ImaskCodeBookCC = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
			cvSet(ImaskCodeBook, cvScalar(255));

			cvNamedWindow("Raw", 1);
			cvNamedWindow("ForegroundCodeBook", 1);
			cvNamedWindow("CodeBook_ConnectComp", 1);
		}

		// If we've got an rawImage and are good to go:
		if (rawImage)
		{
			cvCvtColor(rawImage, yuvImage, CV_BGR2YCrCb);  // YUV For codebook method
			// This is where we build our background model
			if (!pause && nframes - 1 < nframesToLearnBG)
				cvBGCodeBookUpdate(model, yuvImage);

			if (nframes - 1 == nframesToLearnBG)
				cvBGCodeBookClearStale(model, model->t / 2);

			// Find the foreground if any
			if( nframes-1 >= nframesToLearnBG)
			{
				// Find foreground by codebook method
				cvBGCodeBookDiff(model, yuvImage, ImaskCodeBook);
				// This part just to visualize bounding boxes and centers if desired
				cvCopy(ImaskCodeBook, ImaskCodeBookCC);
				cvSegmentFGMask(ImaskCodeBookCC);
			}

			// Display
			cvShowImage("Raw", rawImage);
			cvShowImage("ForegroundCodeBook", ImaskCodeBook);
			cvShowImage("CodeBook_ConnectComp", ImaskCodeBookCC);
		}

		// User input:
		c = cvWaitKey(100) & 0xFF;
		c = tolower(c);
		// End processing on ESC, q or Q
		if (c == 27 || c == 'q')
			break;
		// Else check for user input
		switch (c)
		{
		case 'h':
			//help();
			break;
		case 'p':
			pause = !pause;
			break;
		case 's':
			singlestep = !singlestep;
			pause = false;
			break;
		case 'r':
			pause = false;
			singlestep = false;
			break;
		case ' ':
			cvBGCodeBookClearStale(model, 0);
			nframes = 0;
			break;
			//CODEBOOK PARAMS
		case 'y': case '0':
		case 'u': case '1':
		case 'v': case '2':
		case 'a': case '3':
		case 'b':
			ch[0] = c == 'y' || c == '0' || c == 'a' || c == '3';
			ch[1] = c == 'u' || c == '1' || c == 'a' || c == '3' || c == 'b';
			ch[2] = c == 'v' || c == '2' || c == 'a' || c == '3' || c == 'b';
			std::cout << "CodeBook YUV Channels active: " << ch[0] << ", " << ch[1] << ", " << ch[2] << std::endl;
			break;
		case 'i': //modify max classification bounds (max bound goes higher)
		case 'o': //modify max classification bounds (max bound goes lower)
		case 'k': //modify min classification bounds (min bound goes lower)
		case 'l': //modify min classification bounds (min bound goes higher)
			{
				uchar *ptr = c == 'i' || c == 'o' ? model->modMax : model->modMin;
				for (n = 0; n < NCHANNELS; ++n)
				{
					if (ch[n])
					{
						int v = ptr[n] + (c == 'i' || c == 'l' ? 1 : -1);
						ptr[n] = cv::saturate_cast<uchar>(v);
					}
					std::cout << ptr[n] << ',';
				}
				std::cout << " CodeBook " << (c == 'i' || c == 'o' ? "High" : "Low") << " Side" << std::endl;
			}
			break;
		}
	}

	cvReleaseCapture(&capture);
	cvDestroyWindow("Raw");
	cvDestroyWindow("ForegroundCodeBook");
	cvDestroyWindow("CodeBook_ConnectComp");
}
#endif

void refine_segments(const cv::Mat &img, cv::Mat &mask, cv::Mat &dst)
{
	const int num_iterations = 3;

	cv::Mat temp;
	cv::dilate(mask, temp, cv::Mat(), cv::Point(-1,-1), num_iterations);
	cv::erode(temp, temp, cv::Mat(), cv::Point(-1,-1), num_iterations * 2);
	cv::dilate(temp, temp, cv::Mat(), cv::Point(-1,-1), num_iterations);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(temp, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	dst = cv::Mat::zeros(img.size(), CV_8UC3);
	if (contours.empty())
		return;

	// iterate through all the top-level contours, draw each connected component with its own random color
	int largestComp = 0;
	double maxArea = 0.0;
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		const std::vector<cv::Point> &c = contours[idx];
		const double &area = std::fabs(cv::contourArea(cv::Mat(c)));
		if (area > maxArea)
		{
			maxArea = area;
			largestComp = idx;
		}
	}

	cv::drawContours(dst, contours, largestComp, CV_RGB(255, 0, 0), CV_FILLED, 8, hierarchy);
}

void background_segmentation_by_mog()
{
	const int imageWidth = 640, imageHeight = 480;

	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "fail to open vision sensor" << std::endl;
		return;
	}

	const std::string windowName1("background subtraction - input");
	const std::string windowName2("background subtraction - segmented");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

#if 1
	cv::BackgroundSubtractorMOG bgSubtractor;
#else
	const int history = ;
	const int numMixtures = ;
	const double backgroundRatio = ;
	const double noiseSigma = 10;
	cv::BackgroundSubtractorMOG bgSubtractor(history, numMixtures, backgroundRatio, noiseSigma);
#endif

    bool update_bg_model = true;
	cv::Mat frame, fgMask, segmented_img;
	for (;;)
	{
#if 1
		capture >> frame;
#else
		capture >> frame2;

		if (frame2.cols != imageWidth || frame2.rows != imageHeight)
		{
			//cv::resize(frame2, frame, cv::Size(imageWidth, imageHeight), 0.0, 0.0, cv::INTER_LINEAR);
			cv::pyrDown(frame2, frame);
		}
		else frame = frame2;
#endif

        bgSubtractor(frame, fgMask, update_bg_model ? -1 : 0);

        //cvSegmentFGMask(&(IplImage)fgMask);
		refine_segments(frame, fgMask, segmented_img);

		cv::imshow(windowName1, frame);
		cv::imshow(windowName2, segmented_img);

		const int keycode = cv::waitKey(1);
		if (27 == keycode)
			break;
		else if (' ' == keycode)
		{
			update_bg_model = !update_bg_model;
			std::cout << "learn background is in state = " << update_bg_model << std::endl;
		}
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void change_detection()
{
#if 0
	local::background_segmentation();
	local::change_detection_using_codebook();
#endif

	local::background_segmentation_by_mog();
}

}  // namespace my_opencv
