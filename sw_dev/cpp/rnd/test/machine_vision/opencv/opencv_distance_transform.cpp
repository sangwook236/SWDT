//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>


namespace my_opencv {

void distance_transform()
{
	std::list<std::string> filenames;
/*
	filenames.push_back("./machine_vision_data/opencv/pic1.png");
	filenames.push_back("./machine_vision_data/opencv/pic2.png");
	filenames.push_back("./machine_vision_data/opencv/pic3.png");
	filenames.push_back("./machine_vision_data/opencv/pic4.png");
	filenames.push_back("./machine_vision_data/opencv/pic5.png");
	filenames.push_back("./machine_vision_data/opencv/pic6.png");
	filenames.push_back("./machine_vision_data/opencv/stuff.jpg");
	filenames.push_back("./machine_vision_data/opencv/synthetic_face.png");
	filenames.push_back("./machine_vision_data/opencv/puzzle.png");
	filenames.push_back("./machine_vision_data/opencv/fruits.jpg");
	filenames.push_back("./machine_vision_data/opencv/lena_rgb.bmp");
	filenames.push_back("./machine_vision_data/opencv/hand_01.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_05.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_24.jpg");
*/
	filenames.push_back("./machine_vision_data/opencv/hand_01.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_02.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_03.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_04.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_05.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_06.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_07.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_08.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_09.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_10.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_11.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_12.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_13.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_14.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_15.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_16.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_17.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_18.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_19.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_20.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_21.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_22.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_23.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_24.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_25.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_26.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_27.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_28.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_29.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_30.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_31.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_32.jpg");

	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "image file not found: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			//cv::cvtColor(img, gray, CV_RGB2GRAY);

		const int edgeThreshold = 126;
		gray = gray >= edgeThreshold;  // thresholding

		//
		const int distanceType = CV_DIST_C;  // C/Inf metric
		//const int distanceType = CV_DIST_L1;  // L1 metric
		//const int distanceType = CV_DIST_L2;  // L2 metric
		//const int maskSize = CV_DIST_MASK_3;
		//const int maskSize = CV_DIST_MASK_5;
		const int maskSize = CV_DIST_MASK_PRECISE;
		//const int labelType = cv::DIST_LABEL_CCOMP;
		const int labelType = cv::DIST_LABEL_PIXEL;
		const bool buildVoronoi = false;

		cv::Mat dist32f, labels;
		if (!buildVoronoi)
		{
			// FIXME [check] >>
			cv::Mat dist32f1, dist32f2;
			// distance transform of original image
			cv::distanceTransform(gray, dist32f1, distanceType, maskSize);
			// distance transform of inverted image
			cv::distanceTransform(cv::Scalar::all(255) - gray, dist32f2, distanceType, maskSize);
			cv::max(dist32f1, dist32f2, dist32f);
		}
		else
			cv::distanceTransform(gray, dist32f, labels, distanceType, maskSize, labelType);

		//
		cv::Mat dist;
		if (!buildVoronoi)
		{
			double minVal = 0.0, maxVal = 0.0;
			cv::minMaxLoc(dist32f, &minVal, &maxVal);

			cv::Mat dist_img;
#if 0
			//const double scale = 255.0 / maxVal;
			const double scale = 1.0 / maxVal;
			dist32f.convertTo(dist_img, CV_8UC1, scale, 0);
#else
			//const double scale = 255.0 / (maxVal - minVal);
			const double scale = 1.0 / (maxVal - minVal);
			const double offset = -scale * minVal;
			dist32f.convertTo(dist_img, CV_8UC1, scale, offset);
#endif

			cv::cvtColor(dist_img, dist, CV_GRAY2BGR);
		}
		else
		{
			dist.create(labels.size(), CV_8UC3);
			for (int i = 0; i < labels.rows; ++i)
			{
				const int *ll = (const int *)labels.ptr(i);
				const float *dd = (const float *)dist32f.ptr(i);
				unsigned char *d = (unsigned char *)dist.ptr(i);
				for (int j = 0; j < labels.cols; ++j)
				{
					const bool valid = !(0 == ll[j] || 0 == dd[j]);
					const int r = valid ? std::rand() % 255 + 1 : 0, g = valid ? std::rand() % 255 + 1 : 0, b = valid ? std::rand() % 255 + 1 : 0;
					d[j*3] = (unsigned char)b;
					d[j*3+1] = (unsigned char)g;
					d[j*3+2] = (unsigned char)r;
				}
			}
		}

		//
		cv::imshow("distance transform - input", img);
		cv::imshow("distance transform - result", dist);
		//cv::imshow("distance transform - result", gray);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

void distance_transform_using_edge_info()
{
	std::list<std::string> filenames;
#if 0
	filenames.push_back("./machine_vision_data/opencv/pic1.png");
	filenames.push_back("./machine_vision_data/opencv/pic2.png");
	filenames.push_back("./machine_vision_data/opencv/pic3.png");
	filenames.push_back("./machine_vision_data/opencv/pic4.png");
	filenames.push_back("./machine_vision_data/opencv/pic5.png");
	filenames.push_back("./machine_vision_data/opencv/pic6.png");
	filenames.push_back("./machine_vision_data/opencv/stuff.jpg");
	filenames.push_back("./machine_vision_data/opencv/synthetic_face.png");
	filenames.push_back("./machine_vision_data/opencv/puzzle.png");
	filenames.push_back("./machine_vision_data/opencv/fruits.jpg");
	filenames.push_back("./machine_vision_data/opencv/lena_rgb.bmp");
	filenames.push_back("./machine_vision_data/opencv/hand_01.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_05.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_24.jpg");
#elif 0
	filenames.push_back("./machine_vision_data/opencv/hand_01.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_02.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_03.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_04.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_05.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_06.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_07.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_08.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_09.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_10.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_11.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_12.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_13.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_14.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_15.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_16.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_17.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_18.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_19.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_20.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_21.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_22.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_23.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_24.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_25.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_26.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_27.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_28.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_29.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_30.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_31.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_32.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_33.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_34.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_35.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_36.jpg");
#elif 0
	filenames.push_back("./machine_vision_data/opencv/simple_hand_01.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_02.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_03.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_04.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_05.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_06.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_07.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_08.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_09.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_10.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_11.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_12.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_13.jpg");
#elif 1
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211659.png");
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211705.png");
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211713.png");
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211839.png");
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211842.png");

	std::list<std::string> depth_filenames;
	depth_filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211659.png");
	depth_filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211705.png");
	depth_filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211713.png");
	depth_filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211839.png");
	depth_filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211842.png");
#endif

	for (std::list<std::string>::iterator it = filenames.begin(), depth_it = depth_filenames.begin(); it != filenames.end(); ++it, ++depth_it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "color image file not found: " << *it << std::endl;
			continue;
		}
		const cv::Mat depth_img = cv::imread(*depth_it, CV_LOAD_IMAGE_UNCHANGED);  // CV_16UC1
		if (img.empty())
		{
			std::cout << "depth image file not found: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			//cv::cvtColor(img, gray, CV_RGB2GRAY);

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
			const int kernelSize = 3;
			const bool normalize = true;
			cv::boxFilter(gray.clone(), gray, ddepth, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_DEFAULT);
			//cv::blur(gray.clone(), gray, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), cv::BORDER_DEFAULT);  // use the normalized box filter.
		}
#elif 1
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

		// run the edge detector on grayscale.
		const int lowerEdgeThreshold = 30, upperEdgeThreshold = 50;
		const bool useL2 = true;  // if true, use L2 norm. otherwise, use L1 norm (faster).
		const int apertureSize = 3;  // aperture size for the Sobel() operator.
		cv::Mat edge;
		cv::Canny(gray, edge, lowerEdgeThreshold, upperEdgeThreshold, apertureSize, useL2);

		double minVal, maxVal;
#if 0
		// don't need.

		// thresholding.
		cv::minMaxLoc(edge, &minVal, &maxVal);

		const double threshold_ratio = 0.8;
		const double edgeThreshold = minVal + threshold_ratio * (maxVal - minVal);
		edge = edge >= edgeThreshold;
#endif

		{
			cv::imshow("distance transform - edge", edge);

			cv::Mat tmp;
			cv::minMaxLoc(depth_img, &minVal, &maxVal);
			depth_img.convertTo(tmp, CV_32FC1, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));
			cv::imshow("distance transform - depth", tmp);
		}

		{
			//const int distanceType = CV_DIST_C;  // C/Inf metric
			//const int distanceType = CV_DIST_L1;  // L1 metric
			const int distanceType = CV_DIST_L2;  // L2 metric
			//const int maskSize = CV_DIST_MASK_3;
			//const int maskSize = CV_DIST_MASK_5;
			const int maskSize = CV_DIST_MASK_PRECISE;
			//const int labelType = cv::DIST_LABEL_CCOMP;
			const int labelType = cv::DIST_LABEL_PIXEL;

			cv::Mat dist32f, labels;
			cv::distanceTransform(cv::Scalar::all(255) - edge, dist32f, labels, distanceType, maskSize, labelType);

			{
				cv::minMaxLoc(dist32f, &minVal, &maxVal);
				dist32f.convertTo(dist32f, CV_32FC1, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));

				cv::imshow("distance transform - dt", dist32f);
			}

			cv::Mat edge_labels(labels.size(), labels.type(), cv::Scalar::all(0));
			labels.copyTo(edge_labels, edge);

			cv::Mat depth_labels(labels.size(), labels.type(), cv::Scalar::all(0));
			labels.copyTo(depth_labels, depth_img > 0);

			cv::minMaxLoc(depth_labels, &minVal, &maxVal);

			cv::Mat out_img = img.clone();
			//const int num_labels = cv::countNonZero(edge);
			cv::Point minLoc, maxLoc;
			for (int i = (int)minVal; i <= (int)maxVal; ++i)
			{
				if (cv::countNonZero(i == depth_labels) > 0)
				{
					cv::minMaxLoc(i == edge_labels, NULL, NULL, &minLoc, &maxLoc);
					cv::circle(out_img, maxLoc, 1, CV_RGB(0, 255, 0), CV_FILLED, CV_AA, 0);
				}
			}

			{
				cv::imshow("distance transform - result", out_img);
			}
		}

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
