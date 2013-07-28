//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <iterator>
#include <list>
#include <ctime>


namespace my_opencv {

// [ref] ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void canny(const cv::Mat &gray, const int lowerEdgeThreshold, const int upperEdgeThreshold, const bool useL2, cv::Mat &edge);
void sobel(const cv::Mat &gray, const double thresholdRatio, cv::Mat &gradient);

void segment_motion_using_mhi(const double timestamp, const double mhiTimeDuration, const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, cv::Mat &processed_mhi, cv::Mat &component_label_map, std::vector<cv::Rect> &component_rects);

}  // namespace my_opencv

namespace {
namespace local {

//#define __USE_OBB 1
#define __USE_AABB 1
//#define __USE_BS 1

//#define __USE_CANNY 1
#define __USE_SOBEL 1

void save_ref_hand_image()
{
	std::list<std::string> img_filenames;
	img_filenames.push_back("./machine_vision_data/opencv/hand_detection_ref_01.jpg");
	img_filenames.push_back("./machine_vision_data/opencv/hand_detection_ref_02.jpg");
	img_filenames.push_back("./machine_vision_data/opencv/hand_detection_ref_03.jpg");
	img_filenames.push_back("./machine_vision_data/opencv/hand_detection_ref_04.jpg");
	img_filenames.push_back("./machine_vision_data/opencv/hand_detection_ref_05.jpg");

	const std::string windowName1("hand detection - input");
	const std::string windowName2("hand detection - edge");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	for (std::list<std::string>::const_iterator it = img_filenames.begin(); it != img_filenames.end(); ++it)
	{
		const cv::Mat &in_gray = cv::imread(*it, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat gray;

		//cv::equalizeHist(in_gray, gray);
		gray = in_gray;

		cv::imshow(windowName1, gray);

		cv::Mat edge, edge_img;
#if defined(__USE_CANNY)
		const int lowerEdgeThreshold = 30, upperEdgeThreshold = 50;
		const bool useL2 = true;
		my_opencv::canny(gray, lowerEdgeThreshold, upperEdgeThreshold, useL2, edge);
#elif defined(__USE_SOBEL)
		my_opencv::sobel(gray, -1.0, edge);
#endif

		if (!edge.empty())
		{
			const double thresholdRatio = 0.0;
			double minVal = 0.0, maxVal = 0.0;
			cv::minMaxLoc(edge, &minVal, &maxVal);

			const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
			edge.convertTo(edge_img, CV_8UC1, alpha, beta);
			edge_img.setTo(cv::Scalar::all(0), edge < (minVal + (maxVal - minVal) * thresholdRatio));

			const size_t found = it->find_last_of('.');
			const std::string filename = it->substr(0, found);
			const std::string fileext = it->substr(found);
			cv::imwrite(filename + "_edge" + fileext, edge_img);

			cv::imshow(windowName2, edge_img);
		}

		const int key = cv::waitKey(0);
		if (27 == key) break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

void process_bounding_region(const cv::Mat &ref_edge, const cv::Mat &pts_mat, cv::Mat &img, cv::Mat &processed_img)
{
	// oriented bounding box (OBB)
	const cv::RotatedRect obb(cv::minAreaRect(pts_mat));

	// axis-aligned bounding box (AABB)
	const cv::Rect aabb(cv::boundingRect(pts_mat));

	// bounding sphere
	cv::Point2f center;
	float radius = 0.0f;
	cv::minEnclosingCircle(pts_mat, center, radius);

	//
	cv::Mat mask(img.size(), CV_8UC1, cv::Scalar::all(0));
	{
#if defined(__USE_OBB)
		const int num_pts = 4;
		cv::Point2f vertices[num_pts] = { cv::Point2f(), };
		obb.points(vertices);
		const std::vector<cv::Point> pts(vertices, vertices + num_pts);
		const cv::Point *ptr = (cv::Point *)&(pts[0]);
		cv::fillPoly(mask, (const cv::Point **)&ptr, &num_pts, 1, CV_RGB(255,255,255), CV_AA, 0, cv::Point());
#elif defined(__USE_AABB)
		cv::rectangle(mask, aabb.tl(), aabb.br(), CV_RGB(255,255,255), CV_FILLED, CV_AA, 0);
#elif defined(__USE_BS)
		cv::circle(mask, center, cvRound(radius), CV_RGB(255,255,255), CV_FILLED, CV_AA, 0);
#endif
	}

	//
	{
#if defined(__USE_OBB) || defined(__USE_BS)
#if defined(__USE_CANNY)
		cv::Mat mask_img, gray, edge;
		img.copyTo(mask_img, mask);
		cv::cvtColor(mask_img, gray, CV_BGR2GRAY);

		const int lowerEdgeThreshold = 30, upperEdgeThreshold = 50;
		const bool useL2 = true;
		my_opencv::canny(gray, lowerEdgeThreshold, upperEdgeThreshold, useL2, edge);

		const double thresholdRatio = 0.30;
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(edge, &minVal, &maxVal);

		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		edge.convertTo(processed_img, CV_8UC1, alpha, beta);
		processed_img.setTo(cv::Scalar::all(0), edge < (minVal + (maxVal - minVal) * thresholdRatio));
#elif defined(__USE_SOBEL)
		cv::Mat mask_img, gray, edge;
		img.copyTo(mask_img, mask);
		cv::cvtColor(mask_img, gray, CV_BGR2GRAY);

		my_opencv::sobel(gray, -1.0, edge);

		const double thresholdRatio = 0.05;
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(edge, &minVal, &maxVal);

		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		edge.convertTo(processed_img, CV_8UC1, alpha, beta);
		processed_img.setTo(cv::Scalar::all(0), edge < (minVal + (maxVal - minVal) * thresholdRatio));
#endif
#elif defined(__USE_AABB)
#if defined(__USE_CANNY)
		cv::Mat gray, edge, edge_img;
		cv::cvtColor(mask_img, gray, CV_BGR2GRAY);

		const int lowerEdgeThreshold = 30, upperEdgeThreshold = 50;
		const bool useL2 = true;
		my_opencv::canny(gray, lowerEdgeThreshold, upperEdgeThreshold, useL2, edge);

		const double thresholdRatio = 0.30;
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(edge, &minVal, &maxVal);

		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		edge.convertTo(edge_img, CV_8UC1, alpha, beta);
		edge_img.setTo(cv::Scalar::all(0), edge < (minVal + (maxVal - minVal) * thresholdRatio));

		processed_img = cv::Mat::zeros(img.size(), CV_8UC1);
#if defined(__GNUC__)
        {
            cv::Mat processed_img_tmp(processed_img, aabb);
			edge_img.copyTo(processed_img_tmp);
        }
#else
		edge_img.copyTo(processed_img(aabb));
#endif
#elif defined(__USE_SOBEL)
		cv::Mat gray, edge, edge_img;
		cv::cvtColor(img(aabb), gray, CV_BGR2GRAY);

		my_opencv::sobel(gray, -1.0, edge);

		const double thresholdRatio = 0.05;
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(edge, &minVal, &maxVal);

		const double alpha = 255.0 / (maxVal - minVal), beta = -alpha * minVal;
		edge.convertTo(edge_img, CV_8UC1, alpha, beta);
		edge_img.setTo(cv::Scalar::all(0), edge < (minVal + (maxVal - minVal) * thresholdRatio));

		processed_img = cv::Mat::zeros(img.size(), CV_8UC1);
#if defined(__GNUC__)
        {
            cv::Mat processed_img_tmp(processed_img, aabb);
            edge_img.copyTo(processed_img_tmp);
        }
#else
		edge_img.copyTo(processed_img(aabb));
#endif
#endif
#else
		img.copyTo(processed_img, mask);
#endif
	}

	// draw bounding regions
	{
		// oriented bounding box
		cv::Point2f vertices[4] = { cv::Point2f(), };
		obb.points(vertices);
		for (int i = 0; i < 4; ++i)
			cv::line(img, vertices[i], vertices[(i+1)%4], CV_RGB(255,0,0), 1, CV_AA, 0);

		// axis-aligned bounding box
		const cv::Rect aabb(cv::boundingRect(pts_mat));
		cv::rectangle(img, aabb.tl(), aabb.br(), CV_RGB(0,255,0), 1, CV_AA, 0);

		// bounding sphere
		cv::Point2f center;
		float radius = 0.0f;
		cv::minEnclosingCircle(pts_mat, center, radius);
		cv::circle(img, center, cvRound(radius), CV_RGB(0,0,255), 1, CV_AA, 0);
	}

	// chamfer matching
	{
		std::vector<std::vector<cv::Point> > results;
		std::vector<float> costs;

		const double templScale = 1.0;
		const int maxMatches = 20;
		const double minMatchDistance = 1.0;
		const int padX = 3, padY = 3;
		const int scales = 5;
		const double minScale = 0.6, maxScale = 1.6;
		const double orientationWeight = 0.5;
		const double truncate = 20;
		const int best_matched_idx = cv::chamerMatching(
			// FIXME [modify] >>
			//(cv::Mat &)edge_img, (cv::Mat &)ref_edge, results, costs,
			(cv::Mat &)processed_img, (cv::Mat &)ref_edge, results, costs,
			templScale, maxMatches, minMatchDistance, padX, padY,
			scales, minScale, maxScale, orientationWeight, truncate
		);
		if (best_matched_idx < 0)
		{
			std::cout << "object not found" << std::endl;
			return;
		}

		//
		const std::vector<cv::Point> &pts = results[best_matched_idx];
		for (std::vector<cv::Point>::const_iterator it = pts.begin(); it != pts.end(); ++it)
		{
			if (it->inside(cv::Rect(0, 0, img.cols, img.rows)))
				img.at<cv::Vec3b>(*it) = cv::Vec3b(255, 0, 255);
		}
	}
}

void detect_hand_by_motion()
{
	const int imageWidth = 640, imageHeight = 480;

	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	const std::string ref_imge_filename("./machine_vision_data/opencv/hand_detection_ref_01_edge.jpg");
	//const std::string ref_imge_filename("./machine_vision_data/opencv/hand_detection_ref_02_edge.jpg");
	//const std::string ref_imge_filename("./machine_vision_data/opencv/hand_detection_ref_03_edge.jpg");
	//const std::string ref_imge_filename("./machine_vision_data/opencv/hand_detection_ref_04_edge.jpg");
	//const std::string ref_imge_filename("./machine_vision_data/opencv/hand_detection_ref_05_edge.jpg");

	const cv::Mat &ref_edge0 = cv::imread(ref_imge_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (ref_edge0.empty())
	{
		std::cout << "fail to open reference hand image" << std::endl;
		return;
	}
	cv::Mat ref_edge;
	cv::resize(ref_edge0, ref_edge, cv::Size(), 0.5, 0.5);

	//
	const std::string windowName1("hand detection by motion - input");
	const std::string windowName2("hand detection by motion - MHI");
	const std::string windowName3("hand detection by motion - extracted");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName3, cv::WINDOW_AUTOSIZE);

	//cv::imshow(windowName1, ref_edge);
	//cv::waitKey(0);

	//
	const double MHI_TIME_DURATION = 1.0;

	cv::Mat prevgray, gray, frame, frame2;
	cv::Mat mhi, mhi_img, tmp_img;
	cv::Mat last_silhouette, input_img;
	for (;;)
	{
		const double timestamp = (double)std::clock() / CLOCKS_PER_SEC;  // get current time in seconds

#if 1
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}
#else
		capture >> frame2;
		if (frame2.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		if (frame2.cols != imageWidth || frame2.rows != imageHeight)
		{
			//cv::resize(frame2, frame, cv::Size(imageWidth, imageHeight), 0.0, 0.0, cv::INTER_LINEAR);
			cv::pyrDown(frame2, frame);
		}
		else frame = frame2;
#endif

		cv::cvtColor(frame, gray, CV_BGR2GRAY);
		frame.copyTo(input_img);

		// smoothing
#if 0
		// METHOD #1: down-scale and up-scale the image to filter out the noise.

		{
			cv::Mat tmp;
			cv::pyrDown(gray, tmp);
			cv::pyrUp(tmp, gray);
		}
#elif 0
		// METHOD #2: Gaussian filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int kernelSize = 3;
			const double sigma = 2.0;
			cv::GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
		}
#elif 0
		// METHOD #3: box filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int ddepth = -1;  // the output image depth. -1 to use src.depth().
			const int kernelSize = 5;
			const bool normalize = true;
			cv::boxFilter(gray.clone(), gray, ddepth, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_DEFAULT);
			//cv::blur(gray.clone(), gray, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), cv::BORDER_DEFAULT);  // use the normalized box filter.
		}
#elif 0
		// METHOD #4: bilateral filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int diameter = -1;  // diameter of each pixel neighborhood that is used during filtering. if it is non-positive, it is computed from sigmaSpace.
			const double sigmaColor = 3.0;  // for range filter.
			const double sigmaSpace = 50.0;  // for space filter.
			cv::bilateralFilter(gray.clone(), gray, diameter, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
		}
#else
		// METHOD #5: no filtering.

		//gray = gray;
#endif

		cv::cvtColor(gray, mhi_img, CV_GRAY2BGR);

		if (!prevgray.empty())
		{
			if (mhi.empty())
				mhi.create(gray.rows, gray.cols, CV_32FC1);

			cv::Mat processed_mhi, component_label_map;
			std::vector<cv::Rect> component_rects;
			my_opencv::segment_motion_using_mhi(timestamp, MHI_TIME_DURATION, prevgray, gray, mhi, processed_mhi, component_label_map, component_rects);

			//
			{
				double minVal = 0.0, maxVal = 0.0;
				cv::minMaxLoc(processed_mhi, &minVal, &maxVal);
				minVal = maxVal - 1.5 * MHI_TIME_DURATION;

				const double scale = (255.0 - 1.0) / (maxVal - minVal);
				const double offset = 1.0 - scale * minVal;
				processed_mhi.convertTo(tmp_img, CV_8UC1, scale, offset);

				// TODO [decide] >> want to use it ?
				tmp_img.setTo(cv::Scalar(0), component_label_map == 0);

				cv::cvtColor(tmp_img, mhi_img, CV_GRAY2BGR);
				last_silhouette = processed_mhi >= (timestamp - 1.0e-20);  // last silhouette
				mhi_img.setTo(cv::Scalar(255,0,0), last_silhouette);
			}

			// TODO [check] >> unexpected result
			// it happens that the component areas obtained by MHI disappear in motion, especially when changing motion direction
			if (component_rects.empty())
			{
				//std::cout << "************************************************" << std::endl;
				continue;
			}

			size_t k = 1;
			double min_dist = std::numeric_limits<double>::max();
			cv::Rect selected_rect;
			const double center_x = mhi_img.size().width * 0.5, center_y = mhi_img.size().height * 0.5;
			for (std::vector<cv::Rect>::const_iterator it = component_rects.begin(); it != component_rects.end(); ++it, ++k)
			{
				// reject very small components
				if (it->area() < 100 || it->width + it->height < 100)
					continue;

				// check for the case of little motion
				const size_t count = (size_t)cv::countNonZero((component_label_map == k)(*it));
				if (count < it->width * it->height * 0.05)
					continue;

				cv::rectangle(mhi_img, it->tl(), it->br(), CV_RGB(63, 0, 0), 2, 8, 0);

				const double x = it->x + it->width * 0.5, y = it->y + it->height * 0.5;
				const double dist = (x - center_x)*(x - center_x) + (y - center_y)*(y - center_y);
				if (dist < min_dist)
				{
					min_dist = dist;
					selected_rect = *it;
				}
			}

			if (selected_rect.area() > 0 &&
				(selected_rect.area() <= gray.rows * gray.cols / 2))  // reject too large area
				//selected_rect.area() <= 1.5 * average_area)  // reject too much area variation
			{
				cv::rectangle(mhi_img, selected_rect.tl(), selected_rect.br(), CV_RGB(255, 0, 0), 2, 8, 0);

				//
				{
					const int count = cv::countNonZero(last_silhouette);
					std::vector<cv::Point> points;
					points.reserve(count);

					const unsigned char *pixels = last_silhouette.data;
					for (int r = 0; r < last_silhouette.rows; ++r)
					{
						for (int c = 0; c < last_silhouette.cols; ++c, ++pixels)
						{
							if (*pixels > 0 && selected_rect.contains(cv::Point(c, r)))
								points.push_back(cv::Point(c, r));
						}
					}

					cv::Mat processed_img;
					process_bounding_region(ref_edge, cv::Mat(points), input_img, processed_img);
					cv::imshow(windowName3, processed_img);
				}
			}

			cv::imshow(windowName1, input_img);
			if (!mhi_img.empty())
				cv::imshow(windowName2, mhi_img);
		}

		if (cv::waitKey(1) >= 0)
			break;

		std::swap(prevgray, gray);
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
	cv::destroyWindow(windowName3);
}

// [ref] http://www.andol.info/hci/830.htm
void detect_hand_by_skin_color()
{
	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	// skin color
	const cv::Scalar hsv_min(0, 30, 80, 0);
	const cv::Scalar hsv_max(20, 150, 255, 0);
    //const cv::Scalar hsv_min(0, 45, 0, 0);
    //const cv::Scalar hsv_max(20, 255, 255, 0);
    //const cv::Scalar YCrCb_min(0, 131, 80, 0);
    //const cv::Scalar YCrCb_max(255, 185, 135, 0);

	cv::namedWindow("src");
	cv::namedWindow("hsv-img");
	cv::namedWindow("hsv-mask");

	cv::Mat frame, src_img, hsv_img, hsv_mask;
	while (true)
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		src_img = frame;
		cv::imshow("src", src_img);

		cv::cvtColor(src_img, hsv_img, CV_BGR2HSV);
		cv::imshow("hsv-img", hsv_img);

		cv::inRange(hsv_img, hsv_min, hsv_max, hsv_mask);
		cv::imshow("hsv-mask", hsv_mask);

		if (cv::waitKey(1) >= 0)
			break;
	}

	cv::destroyAllWindows();
}

void detect_and_draw(cv::CascadeClassifier &hand_detector, cv::Mat &img)
{
	// detect hands
	std::vector<cv::Rect> hands;
	const double scaleFactor = 1.1;
	const int minNeighbors = 2;
	hand_detector.detectMultiScale(img, hands, scaleFactor, minNeighbors, CV_HAAR_DO_CANNY_PRUNING, cv::Size(24, 24));

	for (std::size_t i = 0; i < hands.size(); ++i)
	{
		const cv::Point center(cvRound(hands[i].x + hands[i].width * 0.5), cvRound(hands[i].y + hands[i].height * 0.5));
		cv::ellipse(img, center, cv::Size(cvRound(hands[i].width * 0.5), cvRound(hands[i].height * 0.5)), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);
	}

	cv::imshow("result", img);
}

// [ref] http://www.andol.info/hci/2020.htm
// [ref] http://www.andol.info/
void detect_hand_by_haar_cascades_detector()
{
	// [ref] https://github.com/yandol/GstHanddetect
	//const std::string hand_cascade_filename("./machine_vision_data/opencv/gsthanddetect/palm.xml");
	const std::string hand_cascade_filename("./machine_vision_data/opencv/gsthanddetect/fist.xml");

	// load the cascades
	cv::CascadeClassifier hand_detector;
	if (!hand_detector.load(hand_cascade_filename))
	{
		std::cout << "a file not found: " << hand_cascade_filename << std::endl;
		return;
	}

	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	cv::namedWindow("input");
	cv::namedWindow("result");

	cv::Mat frame, gray_img;
	while (true)
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		cv::cvtColor(frame, gray_img, CV_BGR2GRAY);

		cv::imshow("input", frame);

		detect_and_draw(hand_detector, gray_img);

		if (cv::waitKey(1) >= 0)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void hand_detection()
{
	//local::save_ref_hand_image();
	//local::detect_hand_by_motion();

	//local::detect_hand_by_skin_color();
	local::detect_hand_by_haar_cascades_detector();
}

}  // namespace my_opencv
