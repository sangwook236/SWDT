//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

void detect_faces_and_eyes(cv::Mat &frame, cv::CascadeClassifier &face_detector, cv::CascadeClassifier &eye_detector)
{
	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
	cv::equalizeHist(frame_gray, frame_gray);

	// detect faces
	std::vector<cv::Rect> faces;
	face_detector.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

	for (size_t i = 0; i < faces.size(); ++i)
	{
		const cv::Point center(cvRound(faces[i].x + faces[i].width * 0.5), cvRound(faces[i].y + faces[i].height * 0.5));
		cv::ellipse(frame, center, cv::Size(cvRound(faces[i].width * 0.5), cvRound(faces[i].height * 0.5)), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);

		// in each face, detect eyes
		const cv::Mat faceROI = frame_gray(faces[i]);
		std::vector<cv::Rect> eyes;
		eye_detector.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

		for (size_t j = 0; j < eyes.size(); ++j)
		{
			const cv::Point center(cvRound(faces[i].x + eyes[j].x + eyes[j].width * 0.5), cvRound(faces[i].y + eyes[j].y + eyes[j].height * 0.5));
			const int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 4, 8, 0);
		}
	}

	// show what you got
	const std::string window_name = "Capture - Face detection";
	cv::imshow(window_name, frame);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// ${OPENCV_HOME}/sample/cpp/tutorial_code/objectDetection/objectDetection.cpp
// ${OPENCV_HOME}/sample/cpp/tutorial_code/objectDetection/objectDetection2.cpp
void face_detection()
{
	const std::string face_cascade_filename = "./machine_vision_data/opencv/haarcascades/haarcascade_frontalface_alt.xml";  // Haar-like feature
	//const std::string face_cascade_filename = "./machine_vision_data/opencv/lbpcascades/lbpcascade_frontalface.xml";  // LBP feature
	const std::string eyes_cascade_filename = "./machine_vision_data/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

	cv::RNG rng(12345);

	// load the cascades
	cv::CascadeClassifier face_detector;
	if (!face_detector.load(face_cascade_filename))
	{
		std::cout << "Error loading" << std::endl;
		return;
	}

	cv::CascadeClassifier eye_detector;
	if (!eye_detector.load(eyes_cascade_filename))
	{
		std::cout << "Error loading" << std::endl;
		return;
	}

	// read the video stream
	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (capture.isOpened())
	{
		cv::Mat frame;
		while (true)
		{
			capture >> frame;

			// apply the classifier to the frame
			if (!frame.empty())
			{
				local::detect_faces_and_eyes(frame, face_detector, eye_detector);
			}
			else
			{
				std::cout << "--(!) No captured frame -- Break!" << std::endl;
				break;
			}

			const int ch = cv::waitKey(1);
			if ('q' == (char)ch) break;
		}
	}
	else
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}
}

}  // namespace my_opencv
