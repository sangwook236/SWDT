//#include "stdafx.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/saliency/samples/computeSaliency.cpp.
void saliency_detection()
{
	const cv::String saliency_algorithm("SPECTRAL_RESIDUAL");  // Static saliency. 
	//const cv::String saliency_algorithm("FINE_GRAINED");  // Static saliency.
	//const cv::String saliency_algorithm("BinWangApr2014");  // Motion saliency. 
	//const cv::String saliency_algorithm("BING");  // Objectness. 

	if (saliency_algorithm.find("SPECTRAL_RESIDUAL") == 0)  // Static saliency.
	{
		cv::Ptr<cv::saliency::Saliency> saliencyAlgorithm = cv::saliency::StaticSaliencySpectralResidual::create();
		if (nullptr == saliencyAlgorithm)
		{
			std::cerr << "Error in the instantiation of the saliency algorithm." << std::endl;
			return;
		}

		//const std::string img_filepath("../data/machine_vision/objects.jpg");
		const std::string img_filepath("../data/machine_vision/tumblr.jpg");

		//const cv::Mat image = cv::imread(img_filepath, cv::IMREAD_COLOR);
		const cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
		if (image.empty())
		{
			std::cerr << "File not found: " << img_filepath << std::endl;
			return;
		}

		cv::Mat saliencyMap;
		if (saliencyAlgorithm->computeSaliency(image, saliencyMap))
		{
			cv::saliency::StaticSaliencySpectralResidual spec;
			cv::Mat binaryMap;
			spec.computeBinaryMap(saliencyMap, binaryMap);

			cv::imshow("Original Image", image);
			cv::imshow("Saliency Map", saliencyMap);
			cv::imshow("Binary Map", binaryMap);
			cv::waitKey(0);
		}
	}
	else if (saliency_algorithm.find("FINE_GRAINED") == 0)  // Static saliency.
	{
		cv::Ptr<cv::saliency::Saliency> saliencyAlgorithm = cv::saliency::StaticSaliencyFineGrained::create();
		if (nullptr == saliencyAlgorithm)
		{
			std::cerr << "Error in the instantiation of the saliency algorithm." << std::endl;
			return;
		}

		const std::string img_filepath("../data/machine_vision/objects.jpg");
		//const std::string img_filepath("../data/machine_vision/tumblr.jpg");

		const cv::Mat image = cv::imread(img_filepath, cv::IMREAD_COLOR);
		//const cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
		if (image.empty())
		{
			std::cerr << "File not found: " << img_filepath << std::endl;
			return;
		}

		cv::Mat saliencyMap;
		if (saliencyAlgorithm->computeSaliency(image, saliencyMap))
		{
			cv::imshow("Original Image", image);
			cv::imshow("Saliency Map", saliencyMap);
			cv::waitKey(0);
		}
	}
	else if (saliency_algorithm.find("BinWangApr2014") == 0)  // Motion saliency.
	{
		cv::Ptr<cv::saliency::Saliency> saliencyAlgorithm = cv::saliency::MotionSaliencyBinWangApr2014::create();
		if (nullptr == saliencyAlgorithm)
		{
			std::cerr << "Error in the instantiation of the saliency algorithm." << std::endl;
			return;
		}

		// Open the capture.
		const cv::String video_name("../data/machine_vision/opencv/flycap-0001.avi");
		const int start_frame = 0;

		cv::VideoCapture cap;
		cap.open(video_name);
		cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
		if (!cap.isOpened())
		{
			std::cout << "Could not initialize capturing: " << video_name << std::endl;
			return;
		}

		cv::Mat frame;

		cap >> frame;
		if (frame.empty())
		{
			return;
		}

		cv::Mat image;
		frame.copyTo(image);

		//
		//cv::Ptr<cv::Size> size = cv::Ptr<cv::Size>(new cv::Size(image.cols, image.rows));
		saliencyAlgorithm.dynamicCast<cv::saliency::MotionSaliencyBinWangApr2014>()->setImagesize(image.cols, image.rows);
		saliencyAlgorithm.dynamicCast<cv::saliency::MotionSaliencyBinWangApr2014>()->init();

		bool paused = false;
		while (true)
		{
			if (!paused)
			{
				cap >> frame;
				cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

				cv::Mat saliencyMap;
				if (saliencyAlgorithm->computeSaliency(frame, saliencyMap))
				{
					std::cout << "Current frame motion saliency done" << std::endl;
				}

				cv::imshow("Image", frame);
				cv::imshow("Saliency Map", saliencyMap * 255);
			}

			const char c = (char)cv::waitKey(2);
			if ('q' == c)
				break;
			if ('p' == c)
				paused = !paused;
		}
	}
	else if (saliency_algorithm.find("BING") == 0)  // Objectness.
	{
		cv::Ptr<cv::saliency::Saliency> saliencyAlgorithm = cv::saliency::ObjectnessBING::create();
		if (nullptr == saliencyAlgorithm)
		{
			std::cerr << "Error in the instantiation of the saliency algorithm." << std::endl;
			return;
		}

		const cv::String training_path("D:/lib_repo/cpp/rnd/opencv_contrib_github/modules/saliency/samples/ObjectnessTrainedModel");
		if (training_path.empty())
		{
			std::cerr << "Path of trained files missing: " << training_path << std::endl;
			return;
		}
		else
		{
			//const std::string img_filepath("../data/machine_vision/objects.jpg");
			const std::string img_filepath("../data/machine_vision/tumblr.jpg");

			cv::Mat image = cv::imread(img_filepath, cv::IMREAD_COLOR);
			if (image.empty())
			{
				std::cerr << "File not found: " << img_filepath << std::endl;
				return;
			}

			saliencyAlgorithm.dynamicCast<cv::saliency::ObjectnessBING>()->setTrainingPath(training_path);
			saliencyAlgorithm.dynamicCast<cv::saliency::ObjectnessBING>()->setBBResDir("../data/machine_vision/opencv/bing_result");

			std::vector<cv::Vec4i> objectnessBoundingBoxes;
			cv::RNG& rng = cv::theRNG();
			if (saliencyAlgorithm->computeSaliency(image, objectnessBoundingBoxes))
			{
				std::cout << "Objectness done." << std::endl;

#if 0
				for (const auto &box : objectnessBoundingBoxes)
				{
					const cv::Rect rct(cv::Point(box[0], box[1]), cv::Point(box[2], box[3]));
					if (rct.area() > 50000)
						cv::rectangle(image, rct.tl(), rct.br(), CV_RGB(rng(256), rng(256), rng(256)), 1, cv::LINE_AA);
				}

				cv::imshow("Image with objectness regions", image);
				cv::waitKey(0);
#else
				struct
				{
					bool operator()(const cv::Vec4i &lhs, const cv::Vec4i &rhs) const
					{
						return cv::Rect(cv::Point(lhs[0], lhs[1]), cv::Point(lhs[2], lhs[3])).area() > cv::Rect(cv::Point(rhs[0], rhs[1]), cv::Point(rhs[2], rhs[3])).area();
					}
				} areaComparator;
				std::sort(objectnessBoundingBoxes.begin(), objectnessBoundingBoxes.end(), areaComparator);
#if 0
				for (size_t i = 0; i < std::min(200, objectnessBoundingBoxes.size()); ++i)
					cv::rectangle(image, cv::Point(objectnessBoundingBoxes[i][0], objectnessBoundingBoxes[i][1]), cv::Point(objectnessBoundingBoxes[i][2], objectnessBoundingBoxes[i][3]), CV_RGB(rng(256), rng(256), rng(256)), 1, cv::LINE_AA);

				cv::imshow("Image with objectness regions", image);
				cv::waitKey(0);
#else
				size_t idx = 1;
				cv::Mat rgb;
				image.copyTo(rgb);
				for (const auto &box : objectnessBoundingBoxes)
				{
					cv::rectangle(rgb, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]), CV_RGB(rng(256), rng(256), rng(256)), 1, cv::LINE_AA);
					if (0 == idx % 10)
					{
						cv::imshow("Objectness regions", rgb);
						cv::waitKey(100);

						image.copyTo(rgb);
					}
					++idx;
				}
#endif
#endif
			}
		}
	}
}

}  // namespace my_opencv
