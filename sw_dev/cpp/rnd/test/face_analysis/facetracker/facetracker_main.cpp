//include "stdafx.h"
#include <FaceTracker/Tracker.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iostream>


namespace {
namespace local {

void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
	int i, n = shape.rows / 2;
	cv::Point p1, p2;
	cv::Scalar c;

	// draw triangulation.
	c = CV_RGB(0, 0, 0);
	for (i = 0; i < tri.rows; ++i)
	{
		if (visi.at<int>(tri.at<int>(i, 0), 0) == 0 ||
			visi.at<int>(tri.at<int>(i, 1), 0) == 0 ||
			visi.at<int>(tri.at<int>(i, 2), 0) == 0)
			continue;
		p1 = cv::Point(shape.at<double>(tri.at<int>(i, 0), 0), shape.at<double>(tri.at<int>(i, 0) + n, 0));
		p2 = cv::Point(shape.at<double>(tri.at<int>(i, 1), 0), shape.at<double>(tri.at<int>(i, 1) + n, 0));
		cv::line(image, p1, p2, c);
		p1 = cv::Point(shape.at<double>(tri.at<int>(i, 0), 0), shape.at<double>(tri.at<int>(i, 0) + n, 0));
		p2 = cv::Point(shape.at<double>(tri.at<int>(i, 2), 0), shape.at<double>(tri.at<int>(i, 2) + n, 0));
		cv::line(image,p1,p2,c);
		p1 = cv::Point(shape.at<double>(tri.at<int>(i, 2), 0), shape.at<double>(tri.at<int>(i, 2) + n, 0));
		p2 = cv::Point(shape.at<double>(tri.at<int>(i, 1), 0), shape.at<double>(tri.at<int>(i, 1) + n, 0));
		cv::line(image,p1,p2,c);
	}

	// draw connections.
	c = CV_RGB(0, 0, 255);
	for (i = 0; i < con.cols; ++i)
	{
		if (visi.at<int>(con.at<int>(0, i), 0) == 0 || visi.at<int>(con.at<int>(1, i), 0) == 0)
			continue;
		p1 = cv::Point(shape.at<double>(con.at<int>(0, i), 0), shape.at<double>(con.at<int>(0, i) + n, 0));
		p2 = cv::Point(shape.at<double>(con.at<int>(1, i), 0), shape.at<double>(con.at<int>(1, i) + n, 0));
		cv::line(image, p1, p2, c, 1);
	}

	// draw points.
	for (i = 0; i < n; ++i)
	{    
		if (0 == visi.at<int>(i, 0)) continue;
		p1 = cv::Point(shape.at<double>(i, 0), shape.at<double>(i + n, 0));
		c = CV_RGB(255, 0, 0);
		cv::circle(image, p1, 2, c);
	}
}

// [ref] ${FACETRACKER_HOME}/src/exe/face_tracker.cc
void basic_example()
{
	// parse command line arguments.
	const std::string ftFile("./data/face_analysis/facetracker/model/face2.tracker");
	const std::string conFile("./data/face_analysis/facetracker/model/face.con");
	const std::string triFile("./data/face_analysis/facetracker/model/face.tri");
	const bool fcheck = false;  // check for failure
	const double scale = 1.0;  // image scaling
	const int fpd = -1;  // frames/detections
	const bool show = true;

	// set other tracking parameters.
	std::vector<int> wSize1(1);
	wSize1[0] = 7;
	std::vector<int> wSize2(3);
	wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;

	int nIter = 5;
	double clamp = 3, fTol = 0.01; 

	FACETRACKER::Tracker model(ftFile.c_str());
	cv::Mat tri = FACETRACKER::IO::LoadTri(triFile.c_str());
	cv::Mat con = FACETRACKER::IO::LoadCon(conFile.c_str());

	// initialize camera and display window.
	cv::Mat frame, gray, im;
	double fps = 0;

	CvCapture *camera = cvCreateCameraCapture(CV_CAP_ANY);
	if (!camera)
	{
		std::cerr << "vision sensor not found" << std::endl;
		return;
	}

	int64 t1, t0 = cvGetTickCount();
	int fnum = 0;
	cvNamedWindow("Face Tracker", 1);
	std::cout << "Hot keys: " << std::endl
		<< "\t ESC - quit" << std::endl
		<< "\t d   - Redetect" << std::endl;

	// loop until quit (i.e user presses ESC).
	bool failed = true;
	while (true)
	{ 
		// grab image, resize and flip.
		IplImage *I = cvQueryFrame(camera);
		if (!I) continue;
		
		frame = cv::cvarrToMat(I, false);
		if (1 == scale) im = frame;
		else cv::resize(frame, im, cv::Size(scale * frame.cols, scale * frame.rows));

		cv::flip(im, im, 1);
		cv::cvtColor(im, gray, CV_BGR2GRAY);

		// track this image.
		std::vector<int> wSize;
		if (failed) wSize = wSize2;
		else wSize = wSize1; 
		if (model.Track(gray, wSize, fpd, nIter, clamp, fTol, fcheck) == 0)
		{
			int idx = model._clm.GetViewIdx();
			failed = false;
			Draw(im, model. _shape, con, tri, model._clm._visi[idx]); 
		}
		else
		{
			if (show)
			{
				cv::Mat R(im, cvRect(0, 0, 150, 50));
				R = cv::Scalar(0, 0, 255);
			}
			model.FrameReset();
			failed = true;
		}

		// draw framerate on display image.
		if (fnum >= 9)
		{      
			t1 = cvGetTickCount();
			fps = 10.0 / ((double(t1 - t0) / cvGetTickFrequency()) / 1e+6); 
			t0 = t1;
			fnum = 0;
		}
		else fnum += 1;

		if (show)
		{
			std::ostringstream sstrm;
			sstrm << (int)cvRound(fps) << "frames/sec";
			cv::putText(im, sstrm.str(), cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255));
		}

		// show image and check for user input.
		cv::imshow("Face Tracker", im); 

		const int c = cvWaitKey(10);
		if (27 == c) break;
		else if('d' == char(c)) model.FrameReset();
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_facetracker {

}  // namespace my_facetracker

int facetracker_main(int argc, char *argv[])
{
	local::basic_example();

	return 0;
}
