//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <ctime>


namespace {
namespace local {

bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs)
 {
	FileStorage fs(filename, FileStorage::READ);
	if(!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}

bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params)
{
	FileStorage fs(filename, FileStorage::READ);
	if(!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}

// REF [site] >> https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html
// REF [file] >> ${OPENCV_HOME}/modules/aruco/samples/detect_markers.cpp
void aruco_detection()
{
	cv::VideoCapture inputVideo;
	inputVideo.open(0);

	cv::Mat cameraMatrix, distCoeffs;
	// Camera parameters are read from somewhere.
	readCameraParameters(cameraMatrix, distCoeffs);

	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

	while (inputVideo.grab())
	{
		cv::Mat image;
		inputVideo.retrieve(image);

		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f>> corners;
		cv::aruco::detectMarkers(image, dictionary, corners, ids);

		// If at least one marker detected.
		cv::Mat imageCopy;
		image.copyTo(imageCopy);
		if (ids.size() > 0)
		{
			cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

			std::vector<cv::Vec3d> rvecs, tvecs;
			cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

			// Draw axis for each marker.
			for (int i=0; i < ids.size(); ++i)
				cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
		}

		cv::imshow("out", imageCopy);
		char key = (char)cv::waitKey(waitTime);
		if (key == 27)
			break;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void ar()
{
	local::aruco_detection();
}

}  // namespace my_opencv
