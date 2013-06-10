#include "../gslic_lib/FastImgSeg.h"
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <list>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>


namespace {
namespace local {

// [ref] ${GSLIC_HOME}/testCV/main.cpp
void gslic_sample(std::vector<cv::Mat> &input_images, const SEGMETHOD seg_method, const double seg_weight, const int num_segments)
{
	cv::Mat mask;
	for (std::vector<cv::Mat>::iterator it = input_images.begin(); it != input_images.end(); ++it)
    {
		const int64 start = cv::getTickCount();

		// gSLIC currently only support 4-dimensional image
		unsigned char *imgBuffer = new unsigned char [it->cols * it->rows * 4];
		memset(imgBuffer, 0, sizeof(unsigned char) * it->cols * it->rows * 4);

		unsigned char *ptr = imgBuffer;
		for (int i = 0; i < it->rows; ++i)
			for (int j = 0; j < it->cols; ++j)
			{
				const cv::Vec3b &bgr = it->at<cv::Vec3b>(i, j);

				*ptr++ = bgr[0];
				*ptr++ = bgr[1];
				*ptr++ = bgr[2];
				++ptr;
			}

		//
		{
			FastImgSeg gslic;
			gslic.initializeFastSeg(it->cols, it->rows, num_segments);

			gslic.LoadImg(imgBuffer);
			gslic.DoSegmentation(seg_method, seg_weight);
			gslic.Tool_GetMarkedImg();  // required for display of segmentation boundary

			delete [] imgBuffer;
			imgBuffer = NULL;

			// build superpixel boundaries
			ptr = gslic.markedImg;
			for (int i = 0; i < it->rows; ++i)
				for (int j = 0; j < it->cols; ++j)
				{
#if 0
					const unsigned char b = *ptr++;
					const unsigned char g = *ptr++;
					const unsigned char r = *ptr++;
					it->at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
					++ptr;
#else
					it->at<cv::Vec3b>(i, j) = cv::Vec3b(*ptr, *(ptr + 1), *(ptr + 2));
					ptr += 4;
#endif
				}
		}

		const int64 elapsed = cv::getTickCount() - start;
		const double freq = cv::getTickFrequency();
		const double etime = elapsed * 1000.0 / freq;
		const double fps = freq / elapsed;
		std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;

		// show superpixel boundary
		cv::imshow("superpixels by gSLIC - boundary", *it);

#if 0
		// show superpixel mask
		// segment indexes are stored in FastImgSeg::segMask
		cv::Mat mask_tmp(it->size(), CV_32SC1, (void *)gslic.segMask);
        double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(mask_tmp, &minVal, &maxVal);
		mask_tmp.convertTo(mask, CV_32FC1, 1.0 / maxVal, 0.0);

		cv::imshow("superpixels by gSLIC - mask", mask);
#endif

		//gslic.clearFastSeg();

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_gslic {

void create_superpixel_by_gSLIC(const cv::Mat &input_image, cv::Mat &superpixel_mask, const SEGMETHOD seg_method, const double seg_weight, const int num_segments)
{
	// gSLIC currently only support 4-dimensional image
	unsigned char *imgBuffer = new unsigned char [input_image.cols * input_image.rows * 4];
	memset(imgBuffer, 0, sizeof(unsigned char) * input_image.cols * input_image.rows * 4);

	unsigned char *ptr = imgBuffer;
	for (int i = 0; i < input_image.rows; ++i)
		for (int j = 0; j < input_image.cols; ++j)
		{
			const cv::Vec3b &bgr = input_image.at<cv::Vec3b>(i, j);

			*ptr++ = bgr[0];
			*ptr++ = bgr[1];
			*ptr++ = bgr[2];
			++ptr;
		}

	//
	FastImgSeg gslic;
	gslic.initializeFastSeg(input_image.cols, input_image.rows, num_segments);

	gslic.LoadImg(imgBuffer);
	gslic.DoSegmentation(seg_method, seg_weight);
	//gslic.Tool_GetMarkedImg();  // required for display of segmentation boundary

	delete [] imgBuffer;
	imgBuffer = NULL;

	//
	cv::Mat(input_image.size(), CV_32SC1, (void *)gslic.segMask).copyTo(superpixel_mask);

	//gslic.clearFastSeg();
}

// [ref] FastImgSeg::Tool_GetMarkedImg()
void create_superpixel_boundary(const cv::Mat &superpixel_mask, cv::Mat &superpixel_boundary)
{
	superpixel_boundary = cv::Mat::zeros(superpixel_mask.size(), CV_8UC1);

	for (int i = 1; i < superpixel_mask.rows - 1; ++i)
		for (int j = 1; j < superpixel_mask.cols - 1; ++j)
		{
			const int idx = superpixel_mask.at<int>(i, j);
			if (idx != superpixel_mask.at<int>(i, j - 1) || idx != superpixel_mask.at<int>(i, j + 1) ||
				idx != superpixel_mask.at<int>(i - 1, j) || idx != superpixel_mask.at<int>(i + 1, j))
				superpixel_boundary.at<unsigned char>(i, j) = 255;
		}
}

}  // namespace my_gslic

int gslic_main(int argc, char *argv[])
{
	bool canUseGPU = false;
	try
	{
		if (cv::gpu::getCudaEnabledDeviceCount() > 0)
		{
			canUseGPU = true;
			std::cout << "GPU info:" << std::endl;
			cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
		}
		else
			std::cout << "GPU not found ..." << std::endl;

		{
			std::list<std::string> input_file_list;
#if 0
			input_file_list.push_back("./machine_vision_data/opencv/pic1.png");
			input_file_list.push_back("./machine_vision_data/opencv/pic2.png");
			input_file_list.push_back("./machine_vision_data/opencv/pic3.png");
			input_file_list.push_back("./machine_vision_data/opencv/pic4.png");
			input_file_list.push_back("./machine_vision_data/opencv/pic5.png");
			input_file_list.push_back("./machine_vision_data/opencv/pic6.png");
			input_file_list.push_back("./machine_vision_data/opencv/stuff.jpg");
			input_file_list.push_back("./machine_vision_data/opencv/synthetic_face.png");
			input_file_list.push_back("./machine_vision_data/opencv/puzzle.png");
			input_file_list.push_back("./machine_vision_data/opencv/fruits.jpg");
			input_file_list.push_back("./machine_vision_data/opencv/lena_rgb.bmp");
			input_file_list.push_back("./machine_vision_data/opencv/hand_01.jpg");
			input_file_list.push_back("./machine_vision_data/opencv/hand_05.jpg");
			input_file_list.push_back("./machine_vision_data/opencv/hand_24.jpg");
#elif 1
			input_file_list.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130530T103805.png");
			input_file_list.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023152.png");
			input_file_list.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023346.png");
			input_file_list.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023359.png");
#endif

			//
			const int num_segments = 1200;
			const SEGMETHOD seg_method = XYZ_SLIC;  // SLIC, RGB_SLIC, XYZ_SLIC
			const double seg_weight = 0.3;

#if 0
			std::vector<cv::Mat> input_images;
			input_images.reserve(input_file_list.size());
			for (std::list<std::string>::iterator it = input_file_list.begin(); it != input_file_list.end(); ++it)
			{
				cv::Mat img(cv::imread(*it, CV_LOAD_IMAGE_COLOR));
				if (img.empty())
				{
					std::cout << "fail to load image file: " << *it << std::endl;
					continue;
				}
			}

			local::gslic_sample(input_images, seg_method, seg_weight, num_segments);
#else
			cv::Mat superpixel_mask, superpixel_boundary;
			for (std::list<std::string>::iterator it = input_file_list.begin(); it != input_file_list.end(); ++it)
			{
				cv::Mat input_image(cv::imread(*it, CV_LOAD_IMAGE_COLOR));
				if (input_image.empty())
				{
					std::cout << "fail to load image file: " << *it << std::endl;
					continue;
				}

				{
					const int64 start = cv::getTickCount();

					// superpixel mask consists of segment indexes.
					my_gslic::create_superpixel_by_gSLIC(input_image, superpixel_mask, seg_method, seg_weight, num_segments);
					my_gslic::create_superpixel_boundary(superpixel_mask, superpixel_boundary);

					const int64 elapsed = cv::getTickCount() - start;
					const double freq = cv::getTickFrequency();
					const double etime = elapsed * 1000.0 / freq;
					const double fps = freq / elapsed;
					std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;
				}

#if 0
				// show superpixel mask
				cv::Mat mask;
				double minVal = 0.0, maxVal = 0.0;
				cv::minMaxLoc(superpixel_mask, &minVal, &maxVal);
				superpixel_mask.convertTo(mask, CV_32FC1, 1.0 / maxVal, 0.0);

				cv::imshow("superpixels by gSLIC - mask", mask);
#endif

#if 1
				// show superpixel boundary
				//cv::Mat superpixel_boundary;
				//my_gslic::create_superpixel_boundary(superpixel_mask, superpixel_boundary);

				cv::Mat img(input_image.clone());
				img.setTo(cv::Scalar(0, 0, 255), superpixel_boundary);

				cv::imshow("superpixels by gSLIC - boundary", img);
#endif

				const unsigned char key = cv::waitKey(0);
				if (27 == key)
					break;
			}
#endif

			cv::destroyAllWindows();
		}
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		return 1;
	}

	return 0;
}
