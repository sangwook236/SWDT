//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iterator>
#include <cstdio>


namespace {
namespace local {

int print_help()
{
	std::cout <<
		" Given a list of chessboard images, the number of corners (nx, ny)\n"
		" on the chessboards, and a flag: useCalibrated for \n"
		"   calibrated (0) or\n"
		"   uncalibrated \n"
		"     (1: use cvStereoCalibrate(), 2: compute fundamental\n"
		"         matrix separately) stereo. \n"
		" Calibrate the cameras and display the\n"
		" rectified results along with the computed disparity images.   \n" << std::endl;
	std::cout << "Usage:\n ./stereo_calib -w board_width -h board_height [-nr /*dot not view results*/] <image list XML/YML file>\n" << std::endl;
	return 0;
}

void StereoCalib(const std::vector<std::string> &imagelist, const cv::Size &boardSize, const float squareSize, const int maxScale, const bool useCalibrated, const bool showRectified, const bool displayCorners)
{
	if (imagelist.size() % 2 != 0)
	{
		std::cout << "Error: the image list contains odd (non-even) number of elements" << std::endl;
		return;
	}

	// ARRAY AND VECTOR STORAGE:

	std::vector<std::vector<cv::Point2f> > imagePoints[2];
	std::vector<std::vector<cv::Point3f> > objectPoints;
	cv::Size imageSize;

	int i, j, k, nimages = (int)imagelist.size() / 2;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	std::vector<std::string> goodImageList;

	for (i = j = 0; i < nimages; ++i)
	{
		for (k = 0; k < 2; ++k)
		{
			const std::string &filename = imagelist[i*2+k];
			cv::Mat img = cv::imread(filename, 0);
			if (img.empty())
				break;
			if (imageSize == cv::Size())
				imageSize = img.size();
			else if (img.size() != imageSize)
			{
				std::cout << "The image " << filename << " has the size different from the first image size. Skipping the pair" << std::endl;
				break;
			}
			bool found = false;
			std::vector<cv::Point2f> &corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; ++scale)
			{
				cv::Mat timg;
				if (1 == scale)
					timg = img;
				else
					cv::resize(img, timg, cv::Size(), scale, scale);
				found = cv::findChessboardCorners(timg, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
				if (found)
				{
					if (scale > 1)
					{
						cv::Mat cornersMat(corners);
						cornersMat *= 1.0 / scale;
					}
					break;
				}
			}
			if (displayCorners)
			{
				std::cout << filename << std::endl;
				cv::Mat cimg, cimg1;
				cvtColor(img, cimg, CV_GRAY2BGR);
				cv::drawChessboardCorners(cimg, boardSize, corners, found);
				const double sf = 640.0 / MAX(img.rows, img.cols);
				cv::resize(cimg, cimg1, cv::Size(), sf, sf);
				cv::imshow("corners", cimg1);
				const char c = (char)cv::waitKey(500);
				if (c == 27 || c == 'q' || c == 'Q')  // Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found)
				break;
			cv::cornerSubPix(img, corners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS,30, 0.01));
		}
		if (k == 2)
		{
			goodImageList.push_back(imagelist[i*2]);
			goodImageList.push_back(imagelist[i*2+1]);
			++j;
		}
	}
	std::cout << j << " pairs have been successfully detected." << std::endl;
	nimages = j;
	if (nimages < 2)
	{
		std::cout << "Error: too little pairs to run the calibration" << std::endl;
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; ++i)
	{
		for (j = 0; j < boardSize.height; ++j)
			for (k = 0; k < boardSize.width; ++k)
				objectPoints[i].push_back(cv::Point3f(j*squareSize, k*squareSize, 0));
	}

	std::cout << "Running stereo calibration ..." << std::endl;

	cv::Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = cv::Mat::eye(3, 3, CV_64F);
	cameraMatrix[1] = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat R, T, E, F;

	const double rms = cv::stereoCalibrate(
		objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F,
		cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5),
		CV_CALIB_FIX_ASPECT_RATIO + CV_CALIB_ZERO_TANGENT_DIST + CV_CALIB_SAME_FOCAL_LENGTH + CV_CALIB_RATIONAL_MODEL + CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5
		//CV_CALIB_USE_INTRINSIC_GUESS
	);
	std::cout << "done with RMS error=" << rms << std::endl;

	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly includes all the output information,
	// we can check the quality of calibration using the epipolar geometry constraint: m2^t*F*m1=0
	double err = 0;
	int npoints = 0;
	std::vector<cv::Vec3f> lines[2];
	for (i = 0; i < nimages; ++i)
	{
		int npt = (int)imagePoints[0][i].size();
		cv::Mat imgpt[2];
		for (k = 0; k < 2; ++k)
		{
			imgpt[k] = cv::Mat(imagePoints[k][i]);
			cv::undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
			cv::computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
		}
		for (j = 0; j < npt; ++j)
		{
			double errij = std::fabs(imagePoints[0][i][j].x*lines[1][j][0] + imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				std::fabs(imagePoints[1][i][j].x*lines[0][j][0] + imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	std::cout << "average reprojection err = " <<  (err / npoints) << std::endl;

	// save intrinsic parameters
	cv::FileStorage fs("./machine_vision_data/opencv/camera_calibration/stereo_calib_intrinsics.yml", CV_STORAGE_WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] << "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		std::cout << "Error: can not save the intrinsic parameters" << std::endl;

	cv::Mat R1, R2, P1, P2, Q;
	cv::Rect validRoi[2];

	cv::stereoRectify(
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]
	);

	fs.open("./machine_vision_data/opencv/camera_calibration/stereo_calib_extrinsics.yml", CV_STORAGE_WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		std::cout << "Error: can not save the intrinsic parameters" << std::endl;

	// OpenCV can handle left-right or up-down camera arrangements
	bool isVerticalStereo = std::fabs(P2.at<double>(1, 3)) > std::fabs(P2.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	if (!showRectified)
		return;

	cv::Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)
	if (useCalibrated)
	{
		// we already computed everything
	}
	// OR ELSE HARTLEY'S METHOD
	else
	// use intrinsic parameters of each camera, but
	// compute the rectification transformation directly
	// from the fundamental matrix
	{
		std::vector<cv::Point2f> allimgpt[2];
		for (k = 0; k < 2; ++k)
		{
			for (i = 0; i < nimages; ++i)
				std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), std::back_inserter(allimgpt[k]));
		}
		F = cv::findFundamentalMat(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), cv::FM_8POINT, 0, 0);
		cv::Mat H1, H2;
		cv::stereoRectifyUncalibrated(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

		R1 = cameraMatrix[0].inv() * H1 * cameraMatrix[0];
		R2 = cameraMatrix[1].inv() * H2 * cameraMatrix[1];
		P1 = cameraMatrix[0];
		P2 = cameraMatrix[1];
	}

	//Precompute maps for cv::remap()
	cv::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	cv::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	cv::Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width * sf);
		h = cvRound(imageSize.height * sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width * sf);
		h = cvRound(imageSize.height * sf);
		canvas.create(h * 2, w, CV_8UC3);
	}

	for (i = 0; i < nimages; ++i)
	{
		for (k = 0; k < 2; ++k)
		{
			cv::Mat img = cv::imread(goodImageList[i*2+k], 0), rimg, cimg;
			cv::remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
			cv::cvtColor(rimg, cimg, CV_GRAY2BGR);
			cv::Mat canvasPart = !isVerticalStereo ? canvas(cv::Rect(w*k, 0, w, h)) : canvas(cv::Rect(0, h*k, w, h));
			cv::resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
			if (useCalibrated)
			{
				cv::Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf), cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
				cv::rectangle(canvasPart, vroi, cv::Scalar(0,0,255), 3, 8);
			}
		}

		if (!isVerticalStereo)
			for (j = 0; j < canvas.rows; j += 16)
				cv::line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
		else
			for (j = 0; j < canvas.cols; j += 16)
				cv::line(canvas, cv::Point(j, 0), cv::Point(j, canvas.rows), cv::Scalar(0, 255, 0), 1, 8);
		cv::imshow("rectified", canvas);
		char c = (char)cv::waitKey();
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

bool readStringList(const std::string &filename, std::vector<std::string> &l)
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

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// [ref] ${OPENCV_HOME}/samples/cpp/stereo_calib.cpp
void stereo_camera_calibration()
{
#if 0
	cv::Size boardSize;
	const float squareSize = 1.f;  // Set this to your actual square size, [unit]

	std::string imagelistfn;
	for (int i = 1; i < argc; ++i)
	{
		if (std::string(argv[i]) == "-w")
		{
			if (sscanf(argv[++i], "%d", &boardSize.width) != 1 || boardSize.width <= 0)
			{
				std::cout << "invalid board width" << endl;
				local::print_help();
				return;
			}
		}
		else if (std::string(argv[i]) == "-h")
		{
			if (sscanf(argv[++i], "%d", &boardSize.height) != 1 || boardSize.height <= 0)
			{
				std::cout << "invalid board height" << endl;
				local::print_help();
				return;
			}
		}
		else if (std::string(argv[i]) == "-nr")
			showRectified = false;
		else if (std::string(argv[i]) == "--help")
		{
			local::print_help();
			return;
		}
		else if (argv[i][0] == '-')
		{
			std::cout << "invalid option " << argv[i] << endl;
			return;
		}
		else
			imagelistfn = argv[i];
	}

	if (imagelistfn == "")
	{
		imagelistfn = "./machine_vision_data/opencv/camera_calibration/stereo_calib.xml";
		boardSize = cv::Size(9, 6);
	}
	else if (boardSize.width <= 0 || boardSize.height <= 0)
	{
		std::cout << "if you specified XML file with chessboards, you should also specify the board width and height (-w and -h options)" << endl;
		return;
	}
#elif 0
	// [ref] http://blog.martinperis.com/2011/01/opencv-stereo-camera-calibration.html
	const std::string imagelistfn("./machine_vision_data/opencv/camera_calibration/stereo_calib_2.xml");

	const cv::Size boardSize(9, 6);
	const float squareSize = 2.5f;  // Set this to your actual square size, [cm]
#elif 0
	// Kinect IR & RGB images
	//const std::string imagelistfn("./machine_vision_data/opencv/camera_calibration/stereo_calib_3.xml");
	const std::string imagelistfn("./machine_vision_data/opencv/camera_calibration/stereo_calib_4.xml");

	const cv::Size boardSize(7, 5);
	const float squareSize = 10.0f;  // Set this to your actual square size, [cm]
#else
	// [ref] ${OPENCV_HOME}/samples/cpp/stereo_calib.xml
	const std::string imagelistfn("./machine_vision_data/opencv/camera_calibration/stereo_calib.xml");

	const cv::Size boardSize(9, 6);
	const float squareSize = 1.f;  // Set this to your actual square size, [cm]
#endif

	std::vector<std::string> imagelist;
	bool ok = local::readStringList(imagelistfn, imagelist);
	if (!ok || imagelist.empty())
	{
		std::cout << "can not open " << imagelistfn << " or the std::string list is empty" << std::endl;
		local::print_help();
		return;
	}

	const int maxScale = 2;
	const bool showRectified = true;
	const bool useCalibrated = true;
	const bool displayCorners = true;
	local::StereoCalib(imagelist, boardSize, squareSize, maxScale, useCalibrated, showRectified, displayCorners);
}

}  // namespace my_opencv
