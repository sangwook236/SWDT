//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <ctime>


namespace {
namespace local {

const char * usage =
	" \nexample command line for calibration from a live feed.\n"
	"   calibration  -w 4 -h 5 -s 0.025 -o camera.yml -op -oe\n"
	" \n"
	" example command line for calibration from a list of stored images:\n"
	"   imagelist_creator image_list.xml *.png\n"
	"   calibration -w 4 -h 5 -s 0.025 -o camera.yml -op -oe image_list.xml\n"
	" where image_list.xml is the standard OpenCV XML/YAML\n"
	" use imagelist_creator to create the xml or yaml list\n"
	" file consisting of the list of strings, e.g.:\n"
	" \n"
	"<?xml version=\"1.0\"?>\n"
	"<opencv_storage>\n"
	"<images>\n"
	"view000.png\n"
	"view001.png\n"
	"<!-- view002.png -->\n"
	"view003.png\n"
	"view010.png\n"
	"one_extra_view.jpg\n"
	"</images>\n"
	"</opencv_storage>\n";

const char *liveCaptureHelp =
	"When the live video from camera is used as input, the following hot-keys may be used:\n"
	"  <ESC>, 'q' - quit the program\n"
	"  'g' - start capturing images\n"
	"  'u' - switch undistortion on/off\n";

void help()
{
	std::cout << "This is a camera calibration sample.\n"
		"Usage: calibration\n"
		"     -w <board_width>         # the number of inner corners per one of board dimension\n"
		"     -h <board_height>        # the number of inner corners per another board dimension\n"
		"     [-pt <pattern>]          # the type of pattern: chessboard or circles' grid\n"
		"     [-n <number_of_frames>]  # the number of frames to use for calibration\n"
		"                              # (if not specified, it will be set to the number\n"
		"                              #  of board views actually available)\n"
		"     [-d <delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
		"                              # (used only for video capturing)\n"
		"     [-s <squareSize>]        # square size in some user-defined units (1 by default)\n"
		"     [-o <out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
		"     [-op]                    # write detected feature points\n"
		"     [-oe]                    # write extrinsic parameters\n"
		"     [-zt]                    # assume zero tangential distortion\n"
		"     [-a <aspectRatio>]       # fix aspect ratio (fx/fy)\n"
		"     [-p]                     # fix the principal point at the center\n"
		"     [-v]                     # cv::flip the captured images around the horizontal axis\n"
		"     [-V]                     # use a video file, and not an image list, uses\n"
		"                              # [input_data] std::string for the video file name\n"
		"     [-su]                    # show undistorted images after calibration\n"
		"     [input_data]             # input data, one of the following:\n"
		"                              #  - text file with a list of the images of the board\n"
		"                              #    the text file can be generated with imagelist_creator\n"
		"                              #  - name of video file with a video of the board\n"
		"                              # if input_data not specified, a live view from the camera is used\n"
		"\n";

	std::cout << '\n' << usage;
	std::cout << '\n' << liveCaptureHelp;
}

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

double computeReprojectionErrors(
	const std::vector<std::vector<cv::Point3f> > &objectPoints,
	const std::vector<std::vector<cv::Point2f> > &imagePoints,
	const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs,
	const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
	std::vector<float> &perViewErrors)
{
	std::vector<cv::Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); ++i)
	{
		cv::projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		err = norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), CV_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err/n);
		totalErr += err*err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

void calcChessboardCorners(cv::Size boardSize, float squareSize, std::vector<cv::Point3f> &corners, Pattern patternType = CHESSBOARD)
{
	corners.resize(0);

	switch (patternType)
	{
	case CHESSBOARD:
	case CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; ++i)
			for (int j = 0; j < boardSize.width; ++j)
				corners.push_back(cv::Point3f(float(j*squareSize), float(i*squareSize), 0));
		break;

	case ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; ++i)
			for (int j = 0; j < boardSize.width; ++j)
				corners.push_back(cv::Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
		break;

	default:
		CV_Error(CV_StsBadArg, "Unknown pattern type\n");
	}
}

bool runCalibration(const std::vector<std::vector<cv::Point2f> > &imagePoints,
	cv::Size imageSize, cv::Size boardSize, Pattern patternType,
	float squareSize, float aspectRatio,
	int flags, cv::Mat &cameraMatrix, cv::Mat &distCoeffs,
	std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs,
	std::vector<float> &reprojErrs,
	double &totalAvgErr)
{
	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	if (flags & CV_CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = aspectRatio;

	distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

	std::vector<std::vector<cv::Point3f> > objectPoints(1);
	calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

	objectPoints.resize(imagePoints.size(),objectPoints[0]);

	double rms = calibrateCamera(
		objectPoints, imagePoints, imageSize, cameraMatrix,
		distCoeffs, rvecs, tvecs,
		flags | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5
		//flag /*| CV_CALIB_FIX_K3*/ | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5
	);
	std::cout << "RMS error reported by calibrateCamera: " << rms << std::endl;

	const bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}

void saveCameraParams(const std::string &filename,
	cv::Size imageSize, cv::Size boardSize,
	float squareSize, float aspectRatio, int flags,
	const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
	const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs,
	const std::vector<float> &reprojErrs,
	const std::vector<std::vector<cv::Point2f> > &imagePoints,
	double totalAvgErr)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);

	std::time_t tt;
	std::time(&tt);
	struct tm *t2 = std::localtime(&tt);
	char buf[1024];
	std::strftime(buf, sizeof(buf)-1, "%c", t2);

	fs << "calibration_time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << boardSize.width;
	fs << "board_height" << boardSize.height;
	fs << "square_size" << squareSize;

	if (flags & CV_CALIB_FIX_ASPECT_RATIO)
		fs << "aspectRatio" << aspectRatio;

	if (0 != flags)
	{
		sprintf(
			buf, "flags: %s%s%s%s",
			flags & CV_CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CV_CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CV_CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CV_CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : ""
		);
		cvWriteComment(*fs, buf, 0);
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	if (!reprojErrs.empty())
		fs << "per_view_reprojection_errors" << cv::Mat(reprojErrs);

	if (!rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		cv::Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
		for (int i = 0; i < (int)rvecs.size(); ++i)
		{
			cv::Mat r = bigmat(cv::Range(i, i+1), cv::Range(0, 3));
			cv::Mat t = bigmat(cv::Range(i, i+1), cv::Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
			//*.t() is MatExpr (not cv::Mat) so we can use assignment operator
			r = rvecs[i].t();
			t = tvecs[i].t();
		}
		cvWriteComment(*fs, "a set of 6-tuples (rotation std::vector + translation std::vector) for each view", 0);
		fs << "extrinsic_parameters" << bigmat;
	}

	if (!imagePoints.empty())
	{
		cv::Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (int i = 0; i < (int)imagePoints.size(); ++i)
		{
			cv::Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
			cv::Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}
}

static bool readStringList(const std::string &filename, std::vector<std::string> &l)
{
	l.resize(0);
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;

	cv::FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != cv::FileNode::SEQ)
		return false;

	cv::FileNodeIterator it = n.begin(), it_end = n.end();
	for ( ; it != it_end; ++it)
		l.push_back((std::string)*it);

	return true;
}


static bool runAndSave(const std::string &outputFilename,
	const std::vector<std::vector<cv::Point2f> > &imagePoints,
	cv::Size imageSize, cv::Size boardSize, Pattern patternType, float squareSize,
	float aspectRatio, int flags, cv::Mat& cameraMatrix,
	cv::Mat &distCoeffs, bool writeExtrinsics, bool writePoints)
{
	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<float> reprojErrs;
	double totalAvgErr = 0;

	const bool ok = runCalibration(
		imagePoints, imageSize, boardSize, patternType, squareSize,
		aspectRatio, flags, cameraMatrix, distCoeffs,
		rvecs, tvecs, reprojErrs, totalAvgErr
	);
	std::cout << (ok ? "Calibration succeeded" : "Calibration failed") << ". avg reprojection error = " << totalAvgErr << std::endl;

	if (ok)
		saveCameraParams(
			outputFilename, imageSize,
			boardSize, squareSize, aspectRatio,
			flags, cameraMatrix, distCoeffs,
			writeExtrinsics ? rvecs : std::vector<cv::Mat>(),
			writeExtrinsics ? tvecs : std::vector<cv::Mat>(),
			writeExtrinsics ? reprojErrs : std::vector<float>(),
			writePoints ? imagePoints : std::vector<std::vector<cv::Point2f> >(),
			totalAvgErr
		);
	return ok;
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// [ref] ${OPENCV_HOME}/samples/cpp/calibration.cpp
void camera_calibration()
{
#if 0
	const std::string outputFilename("./machine_vision_data/opencv/camera_calibration/camera_calib_data.yml");
	std::string inputFilename;

	cv::Size boardSize;
	float squareSize = 1.f, aspectRatio = 1.f;
	int nframes = 10;
	bool writeExtrinsics = false, writePoints = false;
	int flags = 0;
	bool flipVertical = false;
	bool showUndistorted = false;
	bool videofile = false;
	int delay = 1000;
	int cameraId = 0;
	local::Pattern pattern = local::CHESSBOARD;

	if (argc < 2)
	{
		local::help();
		return;
	}

	for (int i = 1; i < argc; ++i)
	{
		const char* s = argv[i];
		if (strcmp(s, "-w") == 0)
		{
			if (sscanf(argv[++i], "%u", &boardSize.width) != 1 || boardSize.width <= 0)
			{
				std::cerr << "Invalid board width" << std::endl;
				return;
			}
		}
		else if(strcmp(s, "-h") == 0)
		{
			if (sscanf(argv[++i], "%u", &boardSize.height) != 1 || boardSize.height <= 0)
			{
				std::cerr << "Invalid board height" << std::endl;
				return;
			}
		}
		else if (strcmp(s, "-pt") == 0)
		{
			++i;
			if (!strcmp(argv[i], "circles"))
				pattern = local::CIRCLES_GRID;
			else if (!strcmp(argv[i], "acircles"))
				pattern = local::ASYMMETRIC_CIRCLES_GRID;
			else if (!strcmp(argv[i], "chessboard"))
				pattern = local::CHESSBOARD;
			else
			{
				std::cerr << "Invalid pattern type: must be chessboard or circles" << std::endl;
				return;
			}
		}
		else if (strcmp(s, "-s") == 0)
		{
			if (sscanf(argv[++i], "%f", &squareSize) != 1 || squareSize <= 0)
			{
				std::cerr << "Invalid board square width" << std::endl;
				return;
			}
		}
		else if (strcmp(s, "-n") == 0)
		{
			if (sscanf(argv[++i], "%u", &nframes) != 1 || nframes <= 3)
			{
				std::cerr << "Invalid number of images" << std::endl;
				return;
			}
		}
		else if (strcmp(s, "-a") == 0)
		{
			if (sscanf(argv[++i], "%f", &aspectRatio) != 1 || aspectRatio <= 0)
			{
				std::cerr << "Invalid aspect ratio" << std::endl;
				return;
			}
			flags |= CV_CALIB_FIX_ASPECT_RATIO;
		}
		else if (strcmp(s, "-d") == 0)
		{
			if (sscanf(argv[++i], "%u", &delay) != 1 || delay <= 0)
			{
				std::cerr << "Invalid delay" << std::endl;
				return;
			}
		}
		else if (strcmp(s, "-op") == 0)
		{
			writePoints = true;
		}
		else if (strcmp(s, "-oe") == 0)
		{
			writeExtrinsics = true;
		}
		else if (strcmp(s, "-zt") == 0)
		{
			flags |= CV_CALIB_ZERO_TANGENT_DIST;
		}
		else if (strcmp(s, "-p") == 0)
		{
			flags |= CV_CALIB_FIX_PRINCIPAL_POINT;
		}
		else if (strcmp(s, "-v") == 0)
		{
			flipVertical = true;
		}
		else if (strcmp(s, "-V") == 0)
		{
			videofile = true;
		}
		else if (strcmp(s, "-o") == 0)
		{
			outputFilename = argv[++i];
		}
		else if (strcmp(s, "-su") == 0)
		{
			showUndistorted = true;
		}
		else if (s[0] != '-')
		{
			if (isdigit(s[0]))
				sscanf(s, "%d", &cameraId);
			else
				inputFilename = s;
		}
		else
		{
			std::cerr << "Unknown option " << s << std::endl;
			return;
		}
	}
#else
	const std::string outputFilename("./machine_vision_data/opencv/camera_calibration/camera_calib_data.yml");

#if 0
	// [ref] ${OPENCV_HOME}/samples/cpp/stereo_calib.xml
	const std::string inputFilename("./machine_vision_data/opencv/camera_calibration/camera_calib.xml");
	const bool videofile = false;

	const cv::Size boardSize(9, 6);
	const float squareSize = 1.0f;  // Set this to your actual square size, [unit]
	const float aspectRatio = 1.0f;
#elif 0
	// [ref] http://blog.martinperis.com/2011/01/opencv-stereo-camera-calibration.html
	const std::string inputFilename("./machine_vision_data/opencv/camera_calibration/camera_calib_2.xml");
	const bool videofile = false;

	const cv::Size boardSize(9, 6);
	const float squareSize = 25.0f;  // Set this to your actual square size, [mm]
	const float aspectRatio = 1.0f;
#elif 1
	// Kinect RGB images
	const std::string inputFilename("./machine_vision_data/opencv/camera_calibration/camera_calib_3.xml");
	const bool videofile = false;

	const cv::Size boardSize(7, 5);
	const float squareSize = 100.0f;  // Set this to your actual square size, [mm]
	const float aspectRatio = 1.0f;
#else
	const std::string inputFilename;
	const bool videofile = true;

	const cv::Size boardSize(9, 6);
	const float squareSize = 1.0f;  // Set this to your actual square size, [unit]
	const float aspectRatio = 1.0f;
#endif

	const bool writeExtrinsics = false, writePoints = false;
	const int flags = 0;
	//const int flags = CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_FIX_PRINCIPAL_POINT;
	const bool flipVertical = false;
	const bool showUndistorted = true;
	const int delay = 1000;
	const local::Pattern pattern = local::CHESSBOARD;

	int nframes = 10;

	const int cameraId = 0;
#endif

	std::vector<std::string> imageList;
	int mode = local::DETECTION;

	cv::VideoCapture capture;
	if (!inputFilename.empty())
	{
		if (!videofile && local::readStringList(inputFilename, imageList))
			mode = local::CAPTURING;
		else
			capture.open(inputFilename);
	}
	else
		capture.open(cameraId);

	if (!capture.isOpened() && imageList.empty())
	{
		std::cout << "Could not initialize video (" << cameraId << ") capture" << std::endl;
		return;
	}

	if (!imageList.empty())
		nframes = (int)imageList.size();

	if (capture.isOpened())
		std::cout << local::liveCaptureHelp;

	cv::namedWindow("Image View", 1);

	cv::Mat cameraMatrix, distCoeffs;
	std::vector<std::vector<cv::Point2f> > imagePoints;
	bool undistortImage = false;
	cv::Size imageSize;
	std::clock_t prevTimestamp = 0;
	for (int i = 0; ; ++i)
	{
		cv::Mat view, viewGray;
		bool blink = false;

		if (capture.isOpened())
		{
			cv::Mat view0;
			capture >> view0;
			view0.copyTo(view);
		}
		else if (i < (int)imageList.size())
			view = cv::imread(imageList[i], CV_LOAD_IMAGE_COLOR);

		if (!view.data)
		{
			if (imagePoints.size() > 0)
				local::runAndSave(
					outputFilename, imagePoints, imageSize,
					boardSize, pattern, squareSize, aspectRatio,
					flags, cameraMatrix, distCoeffs,
					writeExtrinsics, writePoints
				);
			break;
		}

		imageSize = view.size();

		if (flipVertical)
			cv::flip(view, view, 0);

		std::vector<cv::Point2f> pointbuf;
		cv::cvtColor(view, viewGray, CV_BGR2GRAY);

		bool found;
		switch (pattern)
		{
		case local::CHESSBOARD:
			found = cv::findChessboardCorners(
				view, boardSize, pointbuf,
				CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE
			);
			break;
		case local::CIRCLES_GRID:
			found = cv::findCirclesGrid(view, boardSize, pointbuf);
			break;
		case local::ASYMMETRIC_CIRCLES_GRID:
			found = cv::findCirclesGrid(view, boardSize, pointbuf, cv::CALIB_CB_ASYMMETRIC_GRID);
			break;
		default:
			std::cout << "Unknown pattern type" << std::endl;
			return;
		}

		// improve the found corners' coordinate accuracy
		if (pattern == local::CHESSBOARD && found)
			cv::cornerSubPix(
				viewGray, pointbuf, cv::Size(11, 11),
				cv::Size(-1, -1), cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1)
			);

		if (mode == local::CAPTURING && found &&
			(!capture.isOpened() || std::clock() - prevTimestamp > delay * 1e-3 * CLOCKS_PER_SEC))
		{
			imagePoints.push_back(pointbuf);
			prevTimestamp = std::clock();
			blink = capture.isOpened();
		}

		if (found)
			cv::drawChessboardCorners(view, boardSize, cv::Mat(pointbuf), found);

		std::string msg(mode == local::CAPTURING ? "100/100" : (mode == local::CALIBRATED ? "Calibrated" : "Press 'g' to start"));
		int baseLine = 0;
		cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseLine);
		cv::Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

		if (mode == local::CAPTURING)
		{
			if (undistortImage)
				msg = cv::format("%d/%d Undist", (int)imagePoints.size(), nframes);
			else
				msg = cv::format("%d/%d", (int)imagePoints.size(), nframes);
		}

		cv::putText(
			view, msg, textOrigin, 1, 1,
			mode != local::CALIBRATED ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0)
		);

		if (blink)
			bitwise_not(view, view);

		if (mode == local::CALIBRATED && undistortImage)
		{
			cv::Mat temp = view.clone();
			cv::undistort(temp, view, cameraMatrix, distCoeffs);
		}

		cv::imshow("Image View", view);

		const int key = 0xff & cv::waitKey(capture.isOpened() ? 50 : 500);
		if ((key & 255) == 27)
			break;

		if (key == 'u' && mode == local::CALIBRATED)
			undistortImage = !undistortImage;

		if (capture.isOpened() && key == 'g')
		{
			mode = local::CAPTURING;
			imagePoints.clear();
		}

		if (mode == local::CAPTURING && imagePoints.size() >= (unsigned)nframes)
		{
			if (local::runAndSave(outputFilename, imagePoints, imageSize, boardSize, pattern, squareSize, aspectRatio, flags, cameraMatrix, distCoeffs, writeExtrinsics, writePoints))
				mode = local::CALIBRATED;
			else
				mode = local::DETECTION;

			if (!capture.isOpened())
				break;
		}
	}

	if (!capture.isOpened() && showUndistorted)
	{
		cv::Mat view, rview, map1, map2;
		cv::initUndistortRectifyMap(
			cameraMatrix, distCoeffs, cv::Mat(),
			cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
			imageSize, CV_16SC2, map1, map2
		);

		for (std::size_t i = 0; i < imageList.size(); ++i)
		{
			view = cv::imread(imageList[i], CV_LOAD_IMAGE_COLOR);
			if (!view.data) continue;

			//cv::undistort(view, rview, cameraMatrix, distCoeffs, cameraMatrix);
			cv::remap(view, rview, map1, map2, cv::INTER_LINEAR);

			cv::imshow("Image View", rview);

			const int c = 0xff & cv::waitKey();
			if ((c & 255) == 27 || c == 'q' || c == 'Q')
				break;
		}
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
