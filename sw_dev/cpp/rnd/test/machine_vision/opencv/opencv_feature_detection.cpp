//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>
#include <cassert>
#include <cstring>


namespace {
namespace local {
	
void warpPerspectiveRand(const cv::Mat &src, cv::Mat &dst, cv::Mat &H, cv::RNG &rng)
{
	H.create(3, 3, CV_32FC1);
	H.at<float>(0,0) = rng.uniform( 0.8f, 1.2f);
	H.at<float>(0,1) = rng.uniform(-0.1f, 0.1f);
	H.at<float>(0,2) = rng.uniform(-0.1f, 0.1f) * src.cols;
	H.at<float>(1,0) = rng.uniform(-0.1f, 0.1f);
	H.at<float>(1,1) = rng.uniform( 0.8f, 1.2f);
	H.at<float>(1,2) = rng.uniform(-0.1f, 0.1f) * src.rows;
	H.at<float>(2,0) = rng.uniform(-1e-4f, 1e-4f);
	H.at<float>(2,1) = rng.uniform(-1e-4f, 1e-4f);
	H.at<float>(2,2) = rng.uniform( 0.8f, 1.2f);

	cv::warpPerspective(src, dst, H, src.size());
}

// REF [site] >> http://docs.opencv.org/3.0-beta/doc/tutorials/features2d/trackingmotion/harris_detector/harris_detector.html
void harris_corner(const cv::Mat &gray, const cv::Mat &rgb)
{
	const int blockSize = 2;
	const int apertureSize = 3;
	const double k = 0.04;

	// Detect corners.
	cv::Mat dst = cv::Mat::zeros(gray.size(), CV_32FC1);
	cv::cornerHarris(gray, dst, blockSize, apertureSize, k);

	// Normalize.
	cv::Mat dst_norm;
	cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	cv::Mat dst_norm_scaled;
	cv::convertScaleAbs(dst_norm, dst_norm_scaled);

	// Draw a circle around corners.
	const int thresh = 200;  // [0, 255].
	for (int j = 0; j < dst_norm.rows; ++j)
		for (int i = 0; i < dst_norm.cols; ++i)
			if ((int)dst_norm.at<float>(j, i) > thresh)
				cv::circle(dst_norm_scaled, cv::Point(i, j), 5, cv::Scalar::all(0), 2, cv::LINE_8, 0);

	cv::imshow("Harris corner - Original", rgb);
	cv::imshow("Harris corner - Result", dst_norm_scaled);
}

// REF [file] >> ${OPENCV_HOME}/samples/cpp/tutorial_code/TrackingMotion/goodFeaturesToTrack_Demo.cpp
void good_features_to_track(const cv::Mat &rgb, const cv::Mat &gray)
{
	// Parameters for Shi-Tomasi algorithm.
	const double qualityLevel = 0.01;
	const double minDistance = 10.0;
	const int blockSize = 3, gradiantSize = 3;
	const bool useHarrisDetector = false;
	const double k = 0.04;
	const int maxCorners = 23;
	const int winSize = 10;  // Odd number ???

	// Copy the source image.
	cv::Mat dst = rgb.clone();

	// Apply corner detection.
	std::vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(
		gray, corners, maxCorners,
		qualityLevel, minDistance,
		cv::Mat(),
		blockSize, //gradiantSize,
		useHarrisDetector, k
	);

#if 1
	cv::cornerSubPix(
		gray, corners,
		cv::Size(winSize, winSize), cv::Size(-1, -1),
		cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.03)
	);
#endif

	// Draw corners detected.
	std::cout << "Number of corners detected: " << corners.size() << std::endl;
	cv::RNG rng(12345);
	const int radius = 4;
	for (size_t i = 0; i < corners.size(); ++i)
		cv::circle(dst, corners[i], radius, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);

	// Show results.
	cv::imshow("Good features to track - Original", rgb);
	cv::imshow("Good features to track - Result", dst);
}

// REF [file] >> ${OPENCV_HOME}/samples/cpp/detect_blob.cpp
void simple_blob(const cv::Mat &rgb)
{
	cv::SimpleBlobDetector::Params params;
	params.thresholdStep = 10.0f;
	params.minThreshold = 10.0f;
	params.maxThreshold = 220.0f;
	params.minRepeatability = 2;
	params.minDistBetweenBlobs = 10.0f;
	params.filterByColor = false;
	params.blobColor = 0;  // blobColor = 0 to extract dark blobs. blobColor = 255 to extract light blobs.
	params.filterByArea = false;
	params.minArea = 25.0f;
	params.maxArea = 5000.0f;
	params.filterByCircularity = false;
	params.minCircularity = 0.9f;
	params.maxCircularity = 1e37f;
	params.filterByInertia = false;
	params.minInertiaRatio = 0.1f;
	params.maxInertiaRatio = 1e37f;
	params.filterByConvexity = false;
	params.minConvexity = 0.95f;
	params.maxConvexity = 1e37f;

#if 1
	params.filterByArea = true;
	params.minArea = 500.0f;
	params.maxArea = 3000.0f;
#elif 0
	params.filterByCircularity = true;
#elif 0
	params.filterByInertia = true;
	params.minInertiaRatio = 0.0f;
	params.maxInertiaRatio = 0.2f;
#elif 0
	params.filterByConvexity = true;
	params.minConvexity = 0.0f;
	params.maxConvexity = 0.9f;
#elif 0
	params.filterByColor = true;
	params.blobColor = 0;
#endif

	// Color palette.
	std::vector<cv::Vec3b> palette;
	for (int i = 0; i < 65536; ++i)
		palette.push_back(cv::Vec3b((uchar)std::rand(), (uchar)std::rand(), (uchar)std::rand()));

	cv::Ptr<cv::SimpleBlobDetector> sbd(cv::SimpleBlobDetector::create(params));

	//
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::Rect> zone;
	cv::Mat result(rgb.rows, rgb.cols, CV_8UC3);
	sbd->detect(rgb, keypoints, cv::Mat());

	//
	cv::drawKeypoints(rgb, keypoints, result);
	int i = 0;
	for (std::vector<cv::KeyPoint>::iterator kp = keypoints.begin(); kp != keypoints.end(); ++kp, ++i)
		cv::circle(result, kp->pt, (int)kp->size, palette[i % 65536]);

	//
	cv::imshow("Simple Blob - Original", rgb);
	cv::imshow("Simple Blob - Result", result);
}

#if 0
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
#endif

#if 0
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

	CvMSERParams params = cvMSERParams(); //cvMSERParams(5, 60, cvRound(0.2 * grayImage->width * grayImage->height), 0.25, 0.2);
	CvMemStorage *storage= cvCreateMemStorage();
	CvSeq *contours = NULL;
	double t = (double)cvGetTickCount();
	cvExtractMSER(hsv, NULL, &contours, storage, params);
	t = cvGetTickCount() - t;

	cvReleaseImage(&hsv);

	std::cout << "MSER extracted " << contours->total << " contours in " << (t / ((double)cvGetTickFrequency() * 1000.0)) << " ms" << std::endl;

	// Draw MSER with different color.
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

	// Find ellipse (It seems cvFitEllipse2 have error or sth?).
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
#else
// REF [file] >> ${OPENCV_HOME}/samples/cpp/detect_mser.cpp
void mser(const cv::Mat &rgb, const cv::Mat &gray)
{
	const cv::Vec3b bcolors[] =
	{
		cv::Vec3b(0, 0, 255),
		cv::Vec3b(0, 128, 255),
		cv::Vec3b(0, 255, 255),
		cv::Vec3b(0, 255, 0),
		cv::Vec3b(255, 128, 0),
		cv::Vec3b(255, 255, 0),
		cv::Vec3b(255, 0, 0),
		cv::Vec3b(255, 0, 255),
		cv::Vec3b(255, 255, 255)
	};

	cv::Mat dst, yuv, ellipses;
	cv::cvtColor(rgb, yuv, cv::COLOR_BGR2YCrCb);
	cv::cvtColor(gray, dst, cv::COLOR_GRAY2BGR);
	dst.copyTo(ellipses);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Rect> bboxes;
	{
		const int delta = 5;
		const int min_area = 60;
		const int max_area = 14400;
		const double max_variation = 0.25;
		const double min_diversity = .2;
		const int max_evolution = 200;
		const double area_threshold = 1.01;
		const double min_margin = 0.003;
		const int edge_blur_size = 5;
		cv::Ptr<cv::MSER> mser(cv::MSER::create(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size));

		double t = (double)cv::getTickCount();
		mser->detectRegions(yuv, contours, bboxes);
		t = (double)cv::getTickCount() - t;
		std::cout << "MSER extracted " << contours.size() << " contours in " << (t / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;
	}

	// Draw MSER's with different colors.
	for (int i = (int)contours.size() - 1; i >= 0; --i)
	{
		const std::vector<cv::Point>& r = contours[i];
		for (int j = 0; j < (int)r.size(); ++j)
		{
			const cv::Point &pt = r[j];
			dst.at<cv::Vec3b>(pt) = bcolors[i % 9];
		}

		// Find ellipse (It seems cv::fitEllipse2 have error or sth?).
		cv::RotatedRect &box = cv::fitEllipse(r);

		box.angle = (float)CV_PI / 2 - box.angle;
		cv::ellipse(ellipses, box, cv::Scalar(196, 255, 255), 2);
	}

	cv::imshow("MSER - Original", rgb);
	cv::imshow("MSER - Result", dst);
	cv::imshow("MSER - Ellipses", ellipses);
}
#endif

// REF [file] >> ${OPENCV_HOME}/samples/cpp/detect_blob.cpp
void feature_extractor(const cv::Mat &rgb)
{
	// Color palette.
	std::vector<cv::Vec3b> palette;
	for (int i = 0; i < 65536; ++i)
		palette.push_back(cv::Vec3b((uchar)std::rand(), (uchar)std::rand(), (uchar)std::rand()));

	cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::SIFT::create());
	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::SURF::create());
	//cv::Ptr<cv::Feature2D> detector(cv::BRISK::create());
	//cv::Ptr<cv::Feature2D> detector(cv::ORB::create());
	//cv::Ptr<cv::Feature2D> detector(cv::MSER::create());  // REF [function] >> mser(). Use cv::MSER::detectRegions().
	//cv::Ptr<cv::Feature2D> detector(cv::FastFeatureDetector::create());
	//cv::Ptr<cv::Feature2D> detector(cv::AgastFeatureDetector::create());
	//cv::Ptr<cv::Feature2D> detector(cv::GFTTDetector::create());  // REF [function] >> good_features_to_track().
	//cv::Ptr<cv::Feature2D> detector(cv::SimpleBlobDetector::create());  // REF [function] >> simple_blob().
	//cv::Ptr<cv::Feature2D> detector(cv::KAZE::create()), descriptor(detector);
	//cv::Ptr<cv::Feature2D> detector(cv::AKAZE::create()), descriptor(detector);
	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::StarDetector::create());
	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::MSDDetector::create());
	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::HarrisLaplaceFeatureDetector::create());

	//
	const size_t MAX_KEYPOINT_COUNT = 50;

	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::Rect> zone;
	cv::Mat result(rgb.rows, rgb.cols, CV_8UC3);

	std::cout << "Detecting keypoints ..." << std::endl;
	detector->detect(rgb, keypoints, cv::Mat());
	std::cout << '\t' << keypoints.size() << " points detected." << std::endl;
	cv::KeyPointsFilter::retainBest(keypoints, MAX_KEYPOINT_COUNT);
	std::cout << '\t' << keypoints.size() << " points filtered." << std::endl;

	//
	cv::drawKeypoints(rgb, keypoints, result);
	int i = 0;
	for (std::vector<cv::KeyPoint>::iterator kp = keypoints.begin(); kp != keypoints.end(); ++kp, ++i)
		cv::circle(result, kp->pt, (int)kp->size, palette[i % 65536]);

	//
	cv::imshow("Feature detection - Original", rgb);
	cv::imshow("Feature detection - Result", result);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void feature_detection()
{
	std::list<std::string> filenames;

#if 0
	filenames.push_back("../data/machine_vision/opencv/osp_robot_1.jpg");
	filenames.push_back("../data/machine_vision/opencv/osp_robot_2.jpg");
	filenames.push_back("../data/machine_vision/opencv/osp_robot_3.jpg");
	filenames.push_back("../data/machine_vision/opencv/osp_robot_4.jpg");
	filenames.push_back("../data/machine_vision/opencv/osp_rc_car_1.jpg");
	filenames.push_back("../data/machine_vision/opencv/osp_rc_car_2.jpg");
	filenames.push_back("../data/machine_vision/opencv/osp_rc_car_3.jpg");
#elif 0
	filenames.push_back("../data/machine_vision/opencv/beaver_target.png");
	filenames.push_back("../data/machine_vision/opencv/melon_target.png");
	filenames.push_back("../data/machine_vision/opencv/puzzle.png");
	filenames.push_back("../data/machine_vision/opencv/lena_rgb.bmp");
#elif 0
	filenames.push_back("../data/machine_vision/opencv/hand_01.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_02.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_03.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_04.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_05.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_06.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_07.jpg");  // error occurred !!!
	filenames.push_back("../data/machine_vision/opencv/hand_08.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_09.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_10.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_11.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_12.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_13.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_14.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_15.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_16.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_17.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_18.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_19.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_20.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_21.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_22.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_23.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_24.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_25.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_26.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_27.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_28.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_29.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_30.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_31.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_32.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_33.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_34.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_35.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_36.jpg");
#elif 0
	filenames.push_back("../data/machine_vision/opencv/simple_hand_01.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_02.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_03.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_04.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_05.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_06.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_07.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_08.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_09.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_10.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_11.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_12.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_13.jpg");
#elif 1
	filenames.push_back("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg");
	filenames.push_back("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_1.jpg");
	filenames.push_back("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_2.jpg");
	filenames.push_back("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_3.jpg");
	filenames.push_back("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_4.jpg");
	filenames.push_back("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_5.jpg");
	filenames.push_back("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_6.jpg");
	filenames.push_back("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_7.jpg");
	filenames.push_back("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_8.jpg");
#endif

	//
	const int kernelSize = 7;
	const double sigma = 0.0;
	for (std::list<std::string>::const_iterator cit = filenames.begin(); cit != filenames.end(); ++cit)
    {
		cv::Mat rgb(cv::imread(*cit, cv::IMREAD_COLOR));
		if (rgb.empty())
		{
			std::cout << "Failed to load image file: " << *cit << std::endl;
			continue;
		}

#if 1
		// Blur image.
		cv::GaussianBlur(rgb, rgb, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
#endif

		cv::Mat gray;
		cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

		//local::harris_corner(gray, rgb);
		//local::good_features_to_track(rgb, gray);
		//local::simple_blob(rgb);
		//local::mser(rgb, gray);
		local::feature_extractor(rgb);

		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

void feature_detector_evaluation()
{
	const std::string filename("../data/machine_vision/opencv/melon_target.png");

	//
	const cv::Mat img(cv::imread(filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cout << "Failed to load an image file: " << filename << std::endl;
		return;
	}

	cv::RNG rng = cv::theRNG();

	cv::Mat H12;
	cv::Mat img_warped;
	local::warpPerspectiveRand(img, img_warped, H12, rng);
	if (img_warped.empty())
	{
		std::cout << "Failed to create an image" << std::endl;
		return;
	}

	cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();

	// Detect keypoints.
	std::cout << "Detecting keypoints ..." << std::endl;
	std::vector<cv::KeyPoint> keypoints1;
	detector->detect(img, keypoints1);
	std::cout << '\t' << keypoints1.size() << " points detected." << std::endl;

	std::vector<cv::KeyPoint> keypoints2;
	detector->detect(img_warped, keypoints1);
	std::cout << '\t' << keypoints2.size() << " points detected." << std::endl;

	// Evaluate feature detector.
	std::cout << "Evaluate feature detector ..." << std::endl;
	float repeatability;
	int correspCount;
	cv::evaluateFeatureDetector(img, img_warped, H12, &keypoints1, &keypoints2, repeatability, correspCount);
	std::cout << "\tRepeatability = " << repeatability << std::endl;
	std::cout << "\tCorrespondence Count = " << correspCount << std::endl;
}

}  // namespace my_opencv
