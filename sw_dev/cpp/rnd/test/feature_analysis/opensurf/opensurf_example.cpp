//#include "stdafx.h"
#include "../opensurf_lib/surflib.h"
#include "../opensurf_lib/kmeans.h"
#include <iostream>
#include <ctime>


namespace {
namespace local {

//-------------------------------------------------------

int mainImage()
{
	// Declare Ipoints and other stuff
	IpVec ipts;
	IplImage *img = cvLoadImage("./data/feature_analysis/surf/sf.jpg");

	// Detect and describe interest points in the image
	clock_t start = clock();
	surfDetDes(img, ipts, false, 5, 4, 2, 0.0004f); 
	clock_t end = clock();

	std::cout<< "OpenSURF found: " << ipts.size() << " interest points" << std::endl;
	std::cout<< "OpenSURF took: " << float(end - start) / CLOCKS_PER_SEC  << " seconds" << std::endl;

	// Draw the detected points
	drawIpoints(img, ipts);

	// Display the result
	showImage(img);

	return 0;
}

//-------------------------------------------------------

int mainVideo()
{
	// Initialise capture device
	CvCapture *capture = cvCaptureFromCAM(CV_CAP_ANY);
	if (!capture) error("No Capture");

	// Initialise video writer
	//cv::VideoWriter vw("./data/feature_analysis/surf/out.avi", CV_FOURCC('D','I','V','X'), 10, cvSize(320,240), 1);
	//vw << img;

	// Create a window 
	cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE);

	// Declare Ipoints and other stuff
	IpVec ipts;
	IplImage *img = NULL;

	// Main capture loop
	while (true) 
	{
		// Grab frame from the capture source
		img = cvQueryFrame(capture);

		// Extract surf points
		surfDetDes(img, ipts, false, 4, 4, 2, 0.004f);    

		// Draw the detected points
		drawIpoints(img, ipts);

		// Draw the FPS figure
		drawFPS(img);

		// Display the result
		cvShowImage("OpenSURF", img);

		// If ESC key pressed exit loop
		if ((cvWaitKey(10) & 255) == 27) break;
	}

	cvReleaseCapture(&capture);
	cvDestroyWindow("OpenSURF");

	return 0;
}


//-------------------------------------------------------


int mainMatch()
{
	// Initialise capture device
	CvCapture *capture = cvCaptureFromCAM(CV_CAP_ANY);
	if (!capture) error("No Capture");

	// Declare Ipoints and other stuff
	IpPairVec matches;
	IpVec ipts, ref_ipts;

	// This is the reference object we wish to find in video frame
	// Replace the line below with IplImage *img = cvLoadImage("./data/feature_analysis/surf/object.jpg"); 
	// where object.jpg is the planar object to be located in the video
	IplImage *img = cvLoadImage("./data/feature_analysis/surf/object.jpg"); 
	if (img == NULL) error("Need to load reference image in order to run matching procedure");
	CvPoint src_corners[4] = { {0, 0}, {img->width, 0}, {img->width, img->height}, {0, img->height} };
	CvPoint dst_corners[4];

	// Extract reference object Ipoints
	surfDetDes(img, ref_ipts, false, 3, 4, 3, 0.004f);
	drawIpoints(img, ref_ipts);
	showImage(img);

	// Create a window 
	cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE);

	// Main capture loop
	while (true) 
	{
		// Grab frame from the capture source
		img = cvQueryFrame(capture);

		// Detect and describe interest points in the frame
		surfDetDes(img, ipts, false, 3, 4, 3, 0.004f);

		// Fill match vector
		getMatches(ipts, ref_ipts, matches);

		// This call finds where the object corners should be in the frame
		if (translateCorners(matches, src_corners, dst_corners))
		{
			// Draw box around object
			for (int i = 0; i < 4; ++i)
			{
				const CvPoint &r1 = dst_corners[i % 4];
				const CvPoint &r2 = dst_corners[(i + 1) % 4];
				cvLine(img, cvPoint(r1.x, r1.y), cvPoint(r2.x, r2.y), cvScalar(255, 255, 255), 3);
			}

			for (unsigned int i = 0; i < matches.size(); ++i)
				drawIpoint(img, matches[i].first);
		}

		// Draw the FPS figure
		drawFPS(img);

		// Display the result
		cvShowImage("OpenSURF", img);

		// If ESC key pressed exit loop
		if ((cvWaitKey(10) & 255) == 27) break;
	}

	// Release the capture device
	cvReleaseCapture(&capture);
	cvDestroyWindow("OpenSURF");

	return 0;
}


//-------------------------------------------------------


int mainMotionPoints()
{
	// Initialise capture device
	CvCapture *capture = cvCaptureFromCAM(CV_CAP_ANY);
	if (!capture) error("No Capture");

	// Create a window 
	cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE);

	// Declare Ipoints and other stuff
	IpVec ipts, old_ipts, motion;
	IpPairVec matches;
	IplImage *img = NULL;

	// Main capture loop
	while (true) 
	{
		// Grab frame from the capture source
		img = cvQueryFrame(capture);

		// Detect and describe interest points in the image
		old_ipts = ipts;
		surfDetDes(img, ipts, true, 3, 4, 2, 0.0004f);

		// Fill match vector
		getMatches(ipts,old_ipts, matches);
		for (unsigned int i = 0; i < matches.size(); ++i) 
		{
			const float &dx = matches[i].first.dx;
			const float &dy = matches[i].first.dy;
			float speed = std::sqrt(dx*dx + dy*dy);
			if (speed > 5 && speed < 30) 
				drawIpoint(img, matches[i].first, 3);
		}

		// Display the result
		cvShowImage("OpenSURF", img);

		// If ESC key pressed exit loop
		if ((cvWaitKey(10) & 255) == 27) break;
	}

	// Release the capture device
	cvReleaseCapture(&capture);
	cvDestroyWindow("OpenSURF");

	return 0;
}


//-------------------------------------------------------

int mainStaticMatch()
{
	IplImage *img1 = cvLoadImage("./data/feature_analysis/surf/img1.jpg");
	IplImage *img2 = cvLoadImage("./data/feature_analysis/surf/img2.jpg");

	IpVec ipts1, ipts2;
	surfDetDes(img1, ipts1, false, 4, 4, 2, 0.0001f);
	surfDetDes(img2, ipts2, false, 4, 4, 2, 0.0001f);

	IpPairVec matches;
	getMatches(ipts1, ipts2, matches);

	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		drawPoint(img1, matches[i].first);
		drawPoint(img2, matches[i].second);

		const int &w = img1->width;
		cvLine(img1, cvPoint(matches[i].first.x, matches[i].first.y), cvPoint(matches[i].second.x + w, matches[i].second.y), cvScalar(255, 255, 255), 1);
		cvLine(img2, cvPoint(matches[i].first.x - w, matches[i].first.y), cvPoint(matches[i].second.x, matches[i].second.y), cvScalar(255, 255, 255), 1);
	}

	std::cout<< "Matches: " << matches.size() << std::endl;

	cvNamedWindow("1", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("2", CV_WINDOW_AUTOSIZE);
	cvShowImage("1", img1);
	cvShowImage("2", img2);
	cvWaitKey(0);

	return 0;
}

//-------------------------------------------------------

int mainKmeans()
{
	IplImage *img = cvLoadImage("./data/feature_analysis/surf/img1.jpg");
	IpVec ipts;
	Kmeans km;

	// Get Ipoints
	surfDetDes(img, ipts, true, 3, 4, 2, 0.0006f);

	for (int repeat = 0; repeat < 10; ++repeat)
	{

		IplImage *img = cvLoadImage("./data/feature_analysis/surf/img1.jpg");
		km.Run(&ipts, 5, true);
		drawPoints(img, km.clusters);

		for (unsigned int i = 0; i < ipts.size(); ++i)
		{
			cvLine(img, cvPoint(ipts[i].x, ipts[i].y), cvPoint(km.clusters[ipts[i].clusterIndex].x, km.clusters[ipts[i].clusterIndex].y), cvScalar(255, 255, 255));
		}

		showImage(img);
	}

	return 0;
}

}  // namespace local
}  // unnamed namespace

namespace my_opensurf {

// [ref] ${OPENSURF_HOME}/src/main.cpp
void example()
{
	//-------------------------------------------------------
	// In order to you use OpenSURF, the following illustrates some of the simple tasks you can do.
	// It takes only 1 function call to extract described SURF features!
	// Define PROCEDURE as:
	//  - 1 and supply image path to run on static image
	//  - 2 to capture from a webcam
	//  - 3 to match find an object in an image (work in progress)
	//  - 4 to display moving features (work in progress)
	//  - 5 to show matches between static images
	const int PROCEDURE = 5;

	if (1 == PROCEDURE) local::mainImage();
	if (2 == PROCEDURE) local::mainVideo();
	if (3 == PROCEDURE) local::mainMatch();
	if (4 == PROCEDURE) local::mainMotionPoints();
	if (5 == PROCEDURE) local::mainStaticMatch();
	if (6 == PROCEDURE) local::mainKmeans();
}

}  // namespace my_opensurf
