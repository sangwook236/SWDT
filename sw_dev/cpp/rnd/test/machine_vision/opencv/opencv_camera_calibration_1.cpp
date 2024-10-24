#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


using namespace std::literals::chrono_literals;

namespace {
namespace local {

void find_corners_by_chessboard_corners()
{
#if 0
	const std::string pattern_image_path("path/to/chessboard.png");  // 13 x 14
	const cv::Size pattern_size(13, 14);
#else
	// Generate calibration pattern:
	//	https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
	//
	//	python gen_pattern.py -o chessboard.svg --rows 9 --columns 6 --type checkerboard --square_size 20
	
	const std::string pattern_image_path("./chessboard.png");  // 5 x 8
	const cv::Size pattern_size(5, 8);
#endif

	// Load an image
	cv::Mat img = cv::imread(pattern_image_path, cv::IMREAD_GRAYSCALE);
	if (img.empty())
	{
		std::cerr << "Pattern image file not found, " << pattern_image_path << std::endl;
		return;
	}

#if 0
	cv::Mat img_blurred;
	cv::GaussianBlur(img, img_blurred, cv::Size(5, 5), 0);
#else
	cv::Mat &img_blurred = img;
#endif

	std::cout << "Finding chessboard corners..." << std::endl;
	const auto start_time(std::chrono::high_resolution_clock::now());
	std::vector<cv::Point2f> corners;
	const bool found = cv::findChessboardCorners(img_blurred, pattern_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
	if (!found)
	{
		std::cerr << "Chessboard corners not found." << std::endl;
		return;
	}

	// Refine pixel coordinates for given 2D points
	//const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, DBL_EPSILON);
	const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.001);
	cv::cornerSubPix(img_blurred, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "Chessboard corners found (#corners = " << corners.size() << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

	// Visualize
	cv::Mat img_corners;
	cv::cvtColor(img, img_corners, cv::COLOR_GRAY2BGR);
	cv::drawChessboardCorners(img_corners, pattern_size, corners, found);

	cv::Mat img_pts;
	cv::cvtColor(img, img_pts, cv::COLOR_GRAY2BGR);
	for (const auto &corner: corners)
		cv::circle(img_pts, corner, 1, cv::Scalar(0, 255, 0));

	cv::imshow("Pattern Image", img);
	cv::imshow("Blurred Pattern Image", img_blurred);
	cv::imshow("Corner Image", img_corners);
	cv::imshow("Corner Point Image", img_pts);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void find_circles_by_circles_grid()
{
#if 0
	const std::string pattern_image_path("/work/inno3d/240930_inno3d_calibration_data/pattern_raw_images/0/Cal2D.bmp");  // 13 x 14
	const cv::Size pattern_size(13, 14);
#else
	// Generate calibration pattern:
	//	https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
	//
	//	python gen_pattern.py -o circleboard.svg --rows 7 --columns 5 --type circles --square_size 15
	//	python gen_pattern.py -o acircleboard.svg --rows 7 --columns 5 --type acircles --square_size 10 --radius_rate 2

	const std::string pattern_image_path("./circleboard.png");  // 5 x 7
	//const std::string pattern_image_path("./acircleboard.png");  // 5 x 7
	const cv::Size pattern_size(5, 7);
#endif

	// Load an image
	cv::Mat img = cv::imread(pattern_image_path, cv::IMREAD_GRAYSCALE);
	if (img.empty())
	{
		std::cerr << "Pattern image file not found, " << pattern_image_path << std::endl;
		return;
	}

#if 0
	cv::Mat img_blurred;
	cv::GaussianBlur(img, img_blurred, cv::Size(5, 5), 0);
#else
	cv::Mat &img_blurred = img;
#endif

	std::cout << "Finding circles grid..." << std::endl;
	const auto start_time(std::chrono::high_resolution_clock::now());
	std::vector<cv::Point2f> centers;
	const bool found = cv::findCirclesGrid(img_blurred, pattern_size, centers, cv::CALIB_CB_SYMMETRIC_GRID, cv::SimpleBlobDetector::create());
	//const bool found = cv::findCirclesGrid(img_blurred, pattern_size, centers, cv::CALIB_CB_ASYMMETRIC_GRID, cv::SimpleBlobDetector::create());
	if (!found)
	{
		std::cerr << "Circles grid not found." << std::endl;
		return;
	}

#if 0
	// TODO [check] >> Does it work?

	// Refine pixel coordinates for given 2D points
	//const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, DBL_EPSILON);
	const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.001);
	cv::cornerSubPix(img_blurred, centers, cv::Size(11, 11), cv::Size(-1, -1), criteria);
#endif
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "Circles grid found (#circles = " << centers.size() << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

	// Visualize
	cv::Mat img_corners;
	cv::cvtColor(img, img_corners, cv::COLOR_GRAY2BGR);
	cv::drawChessboardCorners(img_corners, pattern_size, centers, found);

	cv::Mat img_pts;
	cv::cvtColor(img, img_pts, cv::COLOR_GRAY2BGR);
	for (const auto &center: centers)
		cv::circle(img_pts, center, 1, cv::Scalar(0, 255, 0));

	cv::imshow("Pattern Image", img);
	cv::imshow("Blurred Pattern Image", img_blurred);
	cv::imshow("Center Image", img_corners);
	cv::imshow("Center Point Image", img_pts);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void find_circles_by_contours()
{
#if 0
	const std::string pattern_image_path("/work/inno3d/240930_inno3d_calibration_data/pattern_raw_images/0/Cal2D.bmp");  // 13 x 14
	const cv::Size pattern_size(13, 14);
#else
	// Generate calibration pattern:
	//	https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
	//
	//	python gen_pattern.py -o circleboard.svg --rows 7 --columns 5 --type circles --square_size 15
	//	python gen_pattern.py -o acircleboard.svg --rows 7 --columns 5 --type acircles --square_size 10 --radius_rate 2

	const std::string pattern_image_path("./circleboard.png");  // 5 x 7
	//const std::string pattern_image_path("./acircleboard.png");  // 5 x 7
	const cv::Size pattern_size(5, 7);
#endif

	// Load an image
	cv::Mat img = cv::imread(pattern_image_path, cv::IMREAD_GRAYSCALE);
	if (img.empty())
	{
		std::cerr << "Pattern image file not found, " << pattern_image_path << std::endl;
		return;
	}

#if 0
	cv::Mat img_blurred;
	cv::GaussianBlur(img, img_blurred, cv::Size(5, 5), 0);
#else
	cv::Mat &img_blurred = img;
#endif

	std::cout << "Finding contours..." << std::endl;
	const auto start_time(std::chrono::high_resolution_clock::now());
	std::vector<std::vector<cv::Point>> contours;
	//std::vector<cv::Vec4i> hierarchy;
	//cv::findContours(img_blurred, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(img_blurred, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

#if 0
	std::vector<std::vector<cv::Point>> contours_approx;
	for (const auto &contour: contours)
	{
		std::vector<cv::Point> ctr;
		cv::approxPolyDP(contour, ctr, 3, true);
		contours_approx.push_back(ctr);
	}
	contours.assign(contours_approx.begin(), contours_approx.end());
#endif
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "Contours found (#contours = " << contours.size() << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

	std::vector<cv::Point2f> centers;
	size_t contour_idx = 0;
	for (const auto &contour: contours)
	{
		// There should be at least 5 points to fit the ellipse
		//if (contour.size() < 5)
		if (contour.size() < 20)
		{
			std::cerr << "#points of " << contour_idx << "-th contour = " << contour.size() << std::endl;
		}
		else
		{
			const auto &rect = cv::fitEllipse(contour);
			if (std::abs(rect.size.width - rect.size.height) > 1)
				std::cerr << contour_idx << "-th contour is an ellipse: (width, height) = (" << rect.size.width << ", " << rect.size.height << ")." << std::endl;

			centers.push_back(rect.center);
		}
		++contour_idx;
	}

	// Visualize
	cv::Mat img_corners;
	cv::cvtColor(img, img_corners, cv::COLOR_GRAY2BGR);
	cv::drawContours(img_corners, contours, -1, cv::Scalar(0, 0, 255), 1, cv::LINE_8);

	cv::Mat img_pts;
	cv::cvtColor(img, img_pts, cv::COLOR_GRAY2BGR);
	for (const auto &center: centers)
		cv::circle(img_pts, center, 1, cv::Scalar(0, 255, 0));

	cv::imshow("Pattern Image", img);
	cv::imshow("Blurred Pattern Image", img_blurred);
	cv::imshow("Contour Image", img_corners);
	cv::imshow("Center Point Image", img_pts);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

// REF [site] >>
//	https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
//	https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
//	https://learnopencv.com/camera-calibration-using-opencv/
void calibrate_camera_with_chessboards()
{
	// Generate calibration pattern:
	//	https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
	//
	//	python gen_pattern.py -o chessboard.svg --rows 10 --columns 7 --type checkerboard --square_size 20

	// Chessboard images:
	//	https://github.com/opencv/opencv/tree/4.x/samples/data/left01.jpg ~ left14.jpg

	const cv::Size pattern_size(6, 9);  // 6 x 9
	const float square_size(20.0f);

	std::vector<cv::Point3f> corner_points_3d_gt;  // 3D points in the world coordinates
	corner_points_3d_gt.reserve(pattern_size.height * pattern_size.width);
	for (int r = 0; r < pattern_size.height; ++r)
		for (int c = 0; c < pattern_size.width; ++c)
			corner_points_3d_gt.push_back(cv::Point3f(float(c) * square_size, float(r) * square_size, 0.0f));

	// Path of the folder containing checkerboard images
	const std::string pattern_images_path("path/to/*.jpg");
	std::vector<cv::String> image_paths;
	cv::glob(pattern_images_path, image_paths, false);
	std::cout << image_paths.size() << " pattern image files loaded." << std::endl;

	// FIXME [implement] >> Sort image paths

	//std::copy(image_paths.begin(), image_paths.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

	//-----
	int image_width = 0, image_height = 0;
	std::vector<std::vector<cv::Point3f> > points_3d;  // 3D points of each checkerboard corner in the world coordinates
	std::vector<std::vector<cv::Point2f> > points_2d;  // 2D points of each checkerboard corner in the image coordinates
	points_3d.reserve(image_paths.size());
	points_2d.reserve(image_paths.size());
	for (int i = 0; i < image_paths.size(); ++i)
	{
		cv::Mat frame = cv::imread(image_paths[i]);
		if (frame.empty())
		{
			std::cerr << "Pattern image file not found, " << image_paths[i] << std::endl;
			continue;
		}
		if (image_width == 0 || image_height == 0)
		{
			image_width = frame.cols;
			image_height = frame.rows;
		}
		else if (image_width != frame.cols || image_height != frame.rows)
		{
			std::cerr << "Unmatched image size: (" << image_width << ", " << image_height << ") != (" << frame.cols << ", " << frame.rows << ")." << std::endl;
		}

		cv::Mat gray;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// Find chessboard corners
		// If desired number of corners are found in the image then found = true  
		std::vector<cv::Point2f> corners;
		const bool found = cv::findChessboardCorners(gray, pattern_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		// If desired number of corners is detected, we refine the pixel coordinates and display them on the images of chessboard
		if (found)
		{
			// Refine pixel coordinates for given 2D points
			//const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, DBL_EPSILON);
			const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.001);
			cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);

			// Display the detected corner points on the chessboard
			cv::drawChessboardCorners(frame, pattern_size, corners, found);

			points_3d.push_back(corner_points_3d_gt);
			points_2d.push_back(corners);
		}
		assert(points_3d.size() == points_2d.size());

#if 0
		// Show corners detected in the chessboard
		cv::imshow("Image", frame);
		cv::waitKey(0);
#endif
	}
	std::cout << "Image size = (" << image_width << ", " << image_height << ")." << std::endl;

	//-----
	// Calibrate

	cv::Mat cameraMatrix, distCoeffs, rvecs, tvecs;
	{
		// Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern
		//  Perform camera calibration by passing the value of known 3D points (points_3d) and corresponding pixel coordinates of the detected corners (points_2d)
		std::cout << "Calibrating a camera..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, DBL_EPSILON);
		// The overall RMS re-projection error
		//const auto reprojection_error = cv::calibrateCamera(points_3d, points_2d, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria);
		const auto reprojection_error = cv::calibrateCamera(points_3d, points_2d, cv::Size(image_width, image_height), cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A camera calibrated: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;
		if (std::isnan(reprojection_error))
		{
			std::cerr << "Failed to calibrate a camera." << std::endl;
			return;
		}
		std::cout << "RMS re-projection error = " << reprojection_error << std::endl;

		std::cout << "Camera matrix (shape = " << cameraMatrix.size() << "):\n" << cameraMatrix << std::endl;
		std::cout << "Distortion coefficients (shape = " << distCoeffs.size() << "): " << distCoeffs << std::endl;
		std::cout << "Rotation vectors (shape = " << rvecs.size() << "):\n" << rvecs << std::endl;
		std::cout << "Translation vectors (shape = " << tvecs.size() << "):\n" << tvecs << std::endl;
		assert(points_3d.size() == rvecs.rows);
		assert(points_3d.size() == tvecs.rows);
	}

	cv::Rect validPixROI;
	const double alpha = 1.0;  // Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image). See stereoRectify for details.
	const cv::Mat &newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, cv::Size(image_width, image_height), alpha, cv::Size(image_width, image_height), &validPixROI, false);
	std::cout << "Valid pixel ROI: [top, left] = " << validPixROI.tl() << ", [bottom, right] = " << validPixROI.br() << std::endl;

	//-----
	// Undistort

	//for (const auto &image_path: image_paths)
	const auto &image_path = image_paths[11];
	{
		cv::Mat img = cv::imread(image_path);
		if (img.empty())
		{
			std::cerr << "Image file not found, " << image_path << std::endl;
			continue;
		}

		// Undistort (the easiest way)
		cv::Mat img_undistorted;
		{
			std::cout << "Undistorting..." << std::endl;
			const auto start_time(std::chrono::high_resolution_clock::now());
			cv::undistort(img, img_undistorted, cameraMatrix, distCoeffs, newCameraMatrix);
			const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
			std::cout << "Undistorted: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

			cv::Mat img_undistorted_valid = img_undistorted(validPixROI);

			cv::imshow("Image before Undistortion", img);
			cv::imshow("Image after Undistortion", img_undistorted);
			cv::imshow("Image of valid ROI after Undistortion", img_undistorted_valid);
			//cv::imshow("Image Difference", img - img_undistorted);
			cv::waitKey(0);
		}

		// Remap (a little bit more difficult)
		cv::Mat img_remapped;
		{
			std::cout << "Remapping..." << std::endl;
			const auto start_time(std::chrono::high_resolution_clock::now());
			cv::Mat mapx, mapy;
			cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::noArray(), newCameraMatrix, cv::Size(image_width, image_height), CV_32FC1, mapx, mapy);
			cv::remap(img, img_remapped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
			const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
			std::cout << "Remapped: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

			cv::Mat img_remapped_valid = img_remapped(validPixROI);

			cv::imshow("Image before Remapping", img);
			cv::imshow("Image after Remapping", img_remapped);
			cv::imshow("Image of valid ROI after Remapping", img_remapped_valid);
			//cv::imshow("Image Difference", img - img_undistorted);
			cv::waitKey(0);
		}

		//-----
		// Re-projection error

		double sum_squared_errors(0.0);
		double sum_errors(0.0);
		size_t num_pts(0);
		for (auto idx = 0; idx < points_3d.size(); ++idx)
		{
			std::vector<cv::Point2f> img_pts;
			cv::projectPoints(points_3d[idx], rvecs.at<cv::Vec3d>(idx), tvecs.at<cv::Vec3d>(idx), cameraMatrix, distCoeffs, img_pts, cv::noArray(), 0.0);
			assert(img_pts.size() == points_3d[idx].size());
			assert(img_pts.size() == points_2d[idx].size());

			sum_squared_errors += std::pow(cv::norm(points_2d[idx], img_pts, cv::NORM_L2), 2.0);
			sum_errors += cv::norm(points_2d[idx], img_pts, cv::NORM_L2);
			num_pts += img_pts.size();
		}
		std::cout << "RMS = " << (num_pts > 0 ? std::sqrt(sum_squared_errors / num_pts) : 0.0) << std::endl;
		std::cout << "Mean error = " << (num_pts > 0 ? sum_errors / num_pts : 0.0) << std::endl;
	}

	cv::destroyAllWindows();
}

void calibrate_camera_with_circles_grids()
{
	// Generate calibration pattern:
	//	https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
	//
	//	python gen_pattern.py -o circleboard.svg --rows 7 --columns 5 --type circles --square_size 15

	const cv::Size pattern_size(5, 7);  // 5 x 7
	const float square_size(20.0f);

	std::vector<cv::Point3f> center_points_3d_gt;  // 3D points in the world coordinates
	center_points_3d_gt.reserve(pattern_size.height * pattern_size.width);
	for (int r = 0; r < pattern_size.height; ++r)
		for (int c = 0; c < pattern_size.width; ++c)
			center_points_3d_gt.push_back(cv::Point3f(float(c) * square_size, float(r) * square_size, 0.0f));

	// Path of the folder containing circles grid images
	const std::string pattern_images_path("path/to/*.jpg");
	std::vector<cv::String> image_paths;
	cv::glob(pattern_images_path, image_paths, false);
	std::cout << image_paths.size() << " pattern image files loaded." << std::endl;

	// FIXME [implement] >> Sort image paths

	//std::copy(image_paths.begin(), image_paths.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

	//-----
	int image_width = 0, image_height = 0;
	std::vector<std::vector<cv::Point3f> > points_3d;  // 3D points of each circle center in the world coordinates
	std::vector<std::vector<cv::Point2f> > points_2d;  // 2D points of each circle center in the image coordinates
	points_3d.reserve(image_paths.size());
	points_2d.reserve(image_paths.size());
	for (int i = 0; i < image_paths.size(); ++i)
	{
		cv::Mat frame = cv::imread(image_paths[i]);
		if (frame.empty())
		{
			std::cerr << "Pattern image file not found, " << image_paths[i] << std::endl;
			continue;
		}
		if (image_width == 0 || image_height == 0)
		{
			image_width = frame.cols;
			image_height = frame.rows;
		}
		else if (image_width != frame.cols || image_height != frame.rows)
		{
			std::cerr << "Unmatched image size: (" << image_width << ", " << image_height << ") != (" << frame.cols << ", " << frame.rows << ")." << std::endl;
		}

		cv::Mat gray;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// Find centers of the circles
		// If desired number of centers is found in the image then found = true  
		std::vector<cv::Point2f> centers;
		const bool found = cv::findCirclesGrid(gray, pattern_size, centers, cv::CALIB_CB_SYMMETRIC_GRID, cv::SimpleBlobDetector::create());

		// If desired number of centers are detected, we refine the pixel coordinates and display them on the images of circles grid
		if (found)
		{
#if 0
			// TODO [check] >> Does it work?

			// Refine pixel coordinates for given 2D points
			//const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, DBL_EPSILON);
			const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.001);
			cv::cornerSubPix(gray, centers, cv::Size(11, 11), cv::Size(-1, -1), criteria);
#endif

			// Display the detected center points on the circles grid
			cv::drawChessboardCorners(frame, pattern_size, centers, found);

			points_3d.push_back(center_points_3d_gt);
			points_2d.push_back(centers);
		}
		assert(points_3d.size() == points_2d.size());

#if 0
		// Show centers detected in the circles grid
		cv::imshow("Image", frame);
		cv::waitKey(0);
#endif
	}
	std::cout << "Image size = (" << image_width << ", " << image_height << ")." << std::endl;

	//-----
	// Calibrate

	cv::Mat cameraMatrix, distCoeffs, rvecs, tvecs;
	{
		// Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern
		//  Perform camera calibration by passing the value of known 3D points (points_3d) and corresponding pixel coordinates of the detected centers (points_2d)
		std::cout << "Calibrating a camera..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, DBL_EPSILON);
		// The overall RMS re-projection error
		//const auto reprojection_error = cv::calibrateCamera(points_3d, points_2d, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria);
		const auto reprojection_error = cv::calibrateCamera(points_3d, points_2d, cv::Size(image_width, image_height), cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A camera calibrated: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;
		if (std::isnan(reprojection_error))
		{
			std::cerr << "Failed to calibrate a camera." << std::endl;
			return;
		}
		std::cout << "RMS re-projection error = " << reprojection_error << std::endl;

		std::cout << "Camera matrix (shape = " << cameraMatrix.size() << "):\n" << cameraMatrix << std::endl;
		std::cout << "Distortion coefficients (shape = " << distCoeffs.size() << "): " << distCoeffs << std::endl;
		std::cout << "Rotation vectors (shape = " << rvecs.size() << "):\n" << rvecs << std::endl;
		std::cout << "Translation vectors (shape = " << tvecs.size() << "):\n" << tvecs << std::endl;
		assert(points_3d.size() == rvecs.rows);
		assert(points_3d.size() == tvecs.rows);
	}

	cv::Rect validPixROI;
	const double alpha = 1.0;  // Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image). See stereoRectify for details.
	const cv::Mat &newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, cv::Size(image_width, image_height), alpha, cv::Size(image_width, image_height), &validPixROI, false);
	std::cout << "Valid pixel ROI: [top, left] = " << validPixROI.tl() << ", [bottom, right] = " << validPixROI.br() << std::endl;

	//-----
	// Undistort

	//for (const auto &image_path: image_paths)
	const auto &image_path = image_paths[11];
	{
		cv::Mat img = cv::imread(image_path);
		if (img.empty())
		{
			std::cerr << "Image file not found, " << image_path << std::endl;
			continue;
		}

		// Undistort (the easiest way)
		cv::Mat img_undistorted;
		{
			std::cout << "Undistorting..." << std::endl;
			const auto start_time(std::chrono::high_resolution_clock::now());
			cv::undistort(img, img_undistorted, cameraMatrix, distCoeffs, newCameraMatrix);
			const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
			std::cout << "Undistorted: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

			cv::Mat img_undistorted_valid = img_undistorted(validPixROI);

			cv::imshow("Image before Undistortion", img);
			cv::imshow("Image after Undistortion", img_undistorted);
			cv::imshow("Image of valid ROI after Undistortion", img_undistorted_valid);
			cv::imshow("Image Difference", img - img_undistorted);
			cv::waitKey(0);
		}

		// Remap (a little bit more difficult)
		cv::Mat img_remapped;
		{
			std::cout << "Remapping..." << std::endl;
			const auto start_time(std::chrono::high_resolution_clock::now());
			cv::Mat mapx, mapy;
			cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::noArray(), newCameraMatrix, cv::Size(image_width, image_height), CV_32FC1, mapx, mapy);
			cv::remap(img, img_remapped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
			const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
			std::cout << "Remapped: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

			cv::Mat img_remapped_valid = img_remapped(validPixROI);

			cv::imshow("Image before Remapping", img);
			cv::imshow("Image after Remapping", img_remapped);
			cv::imshow("Image of valid ROI after Remapping", img_remapped_valid);
			cv::imshow("Image Difference", img - img_undistorted);
			cv::waitKey(0);
		}

		//-----
		// Re-projection error

		double sum_squared_errors(0.0);
		double sum_errors(0.0);
		size_t num_pts(0);
		for (auto idx = 0; idx < points_3d.size(); ++idx)
		{
			std::vector<cv::Point2f> img_pts;
			cv::projectPoints(points_3d[idx], rvecs.at<cv::Vec3d>(idx), tvecs.at<cv::Vec3d>(idx), cameraMatrix, distCoeffs, img_pts, cv::noArray(), 0.0);
			assert(img_pts.size() == points_3d[idx].size());
			assert(img_pts.size() == points_2d[idx].size());

			sum_squared_errors += std::pow(cv::norm(points_2d[idx], img_pts, cv::NORM_L2), 2.0);
			sum_errors += cv::norm(points_2d[idx], img_pts, cv::NORM_L2);
			num_pts += img_pts.size();
		}
		std::cout << "RMS = " << (num_pts > 0 ? std::sqrt(sum_squared_errors / num_pts) : 0.0) << std::endl;
		std::cout << "Mean error = " << (num_pts > 0 ? sum_errors / num_pts : 0.0) << std::endl;

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void camera_calibration_1()
{
	// REF [site] >>
	//	https://docs.opencv.org/4.x/d6/d55/tutorial_table_of_content_calib3d.html
	//	https://docs.opencv.org/4.x/d9/db7/tutorial_py_table_of_contents_calib3d.html

	// Generate calibration pattern:
	//	https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
	//
	//	python gen_pattern.py -o chessboard.svg --rows 9 --columns 6 --type checkerboard --square_size 20
	//	python gen_pattern.py -o circleboard.svg --rows 7 --columns 5 --type circles --square_size 15
	//	python gen_pattern.py -o acircleboard.svg --rows 7 --columns 5 --type acircles --square_size 10 --radius_rate 2
	//	python gen_pattern.py -o radon_checkerboard.svg --rows 10 --columns 15 --type radon_checkerboard -s 12.1 -m 7 4 7 5 8 5
	//	python gen_pattern.py -o charuco_board.svg --rows 7 --columns 5 -T charuco_board --square_size 30 --marker_size 15 -f DICT_5X5_100.json.gz

	//local::find_corners_by_chessboard_corners();
	//local::find_circles_by_circles_grid();
	//local::find_circles_by_contours();

	local::calibrate_camera_with_chessboards();
	//local::calibrate_camera_with_circles_grids();
}

}  // namespace my_opencv
