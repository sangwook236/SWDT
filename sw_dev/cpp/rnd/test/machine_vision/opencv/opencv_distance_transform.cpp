//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>


namespace my_opencv {

void distance_transform()
{
	std::list<std::string> filenames;
/*
	filenames.push_back("machine_vision_data\\opencv\\pic1.png");
	filenames.push_back("machine_vision_data\\opencv\\pic2.png");
	filenames.push_back("machine_vision_data\\opencv\\pic3.png");
	filenames.push_back("machine_vision_data\\opencv\\pic4.png");
	filenames.push_back("machine_vision_data\\opencv\\pic5.png");
	filenames.push_back("machine_vision_data\\opencv\\pic6.png");
	filenames.push_back("machine_vision_data\\opencv\\stuff.jpg");
	filenames.push_back("machine_vision_data\\opencv\\synthetic_face.png");
	filenames.push_back("machine_vision_data\\opencv\\puzzle.png");
	filenames.push_back("machine_vision_data\\opencv\\fruits.jpg");
	filenames.push_back("machine_vision_data\\opencv\\lena_rgb.bmp");
	filenames.push_back("machine_vision_data\\opencv\\hand_01.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_05.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_24.jpg");
*/
	filenames.push_back("machine_vision_data\\opencv\\hand_01.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_02.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_03.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_04.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_05.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_06.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_07.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_08.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_09.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_10.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_11.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_12.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_13.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_14.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_15.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_16.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_17.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_18.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_19.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_20.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_21.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_22.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_23.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_24.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_25.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_26.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_27.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_28.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_29.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_30.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_31.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_32.jpg");

	const std::string windowName1("distance transform - original");
	const std::string windowName2("distance transform - processed");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			//cv::cvtColor(img, gray, CV_RGB2GRAY);

		const int distanceType = CV_DIST_C;  // C/Inf metric
		//const int distanceType = CV_DIST_L1;  // L1 metric
		//const int distanceType = CV_DIST_L2;  // L2 metric
		//const int maskSize = CV_DIST_MASK_3;
		//const int maskSize = CV_DIST_MASK_5;
		const int maskSize = CV_DIST_MASK_PRECISE;
		const bool buildVoronoi = false;

		const int edgeThreshold = 126;
		gray = gray >= edgeThreshold;  // thresholding

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
			cv::distanceTransform(gray, dist32f, labels, distanceType, maskSize);

		cv::Mat dist;
		if (!buildVoronoi)
		{
			double minVal = 0.0, maxVal = 0.0;
			cv::minMaxLoc(dist32f, &minVal, &maxVal);

			cv::Mat dist8u;
#if 0
			const double scale = 255.0 / maxVal;
			dist32f.convertTo(dist8u, CV_8UC1, scale, 0);
#else
			const double scale = 255.0 / (maxVal - minVal);
			const double offset = -scale * minVal;
			dist32f.convertTo(dist8u, CV_8UC1, scale, offset);
#endif

			cv::cvtColor(dist8u, dist, CV_GRAY2BGR);
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
		cv::imshow(windowName1, img);
		cv::imshow(windowName2, dist);
		//cv::imshow(windowName2, gray);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

}  // namespace my_opencv
