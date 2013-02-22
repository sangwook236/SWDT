//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <iomanip>


namespace {
namespace local {

struct BufferForStereoMatching
{
	cv::gpu::GpuMat imgL, imgR;
	cv::gpu::GpuMat imgDisp;
};

void bm(const std::size_t numDisparities, BufferForStereoMatching &buffer, cv::gpu::Stream &stream = cv::gpu::Stream::Null())
{
	//const int preset = cv::gpu::StereoBM_GPU::BASIC_PRESET;
	const int preset = cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL;
	const int winSize = cv::gpu::StereoBM_GPU::DEFAULT_WINSZ;  // 19
    cv::gpu::StereoBM_GPU bm(preset, numDisparities, winSize);

	const int64 start = cv::getTickCount();

	bm(buffer.imgL, buffer.imgR, buffer.imgDisp, stream);

	const int64 elapsed = cv::getTickCount() - start;
    const double freq = cv::getTickFrequency();
	const double etime = elapsed * 1000.0 / freq;
    const double fps = freq / elapsed;
	std::cout << std::setprecision(4) << "[BM] elapsed time: " << etime <<  ", FPS: " << fps << std::endl;
}

void belief_propagation(const std::size_t numDisparities, BufferForStereoMatching &buffer, cv::gpu::Stream &stream = cv::gpu::Stream::Null())
{
	const int numIterations = cv::gpu::StereoBeliefPropagation::DEFAULT_ITERS;  // 5
	const int numLevels = cv::gpu::StereoBeliefPropagation::DEFAULT_LEVELS;  // 5
	const int msgType = CV_32F;
    cv::gpu::StereoBeliefPropagation bp(numDisparities, numIterations, numLevels, msgType);

	const int64 start = cv::getTickCount();

	bp(buffer.imgL, buffer.imgR, buffer.imgDisp, stream);

	const int64 elapsed = cv::getTickCount() - start;
    const double freq = cv::getTickFrequency();
	const double etime = elapsed * 1000.0 / freq;
    const double fps = freq / elapsed;
	std::cout << std::setprecision(4) << "[BP] elapsed time: " << etime <<  ", FPS: " << fps << std::endl;
}

void constant_space_belief_propagation(const std::size_t numDisparities, BufferForStereoMatching &buffer, cv::gpu::Stream &stream = cv::gpu::Stream::Null())
{
	const int numIterations = cv::gpu::StereoConstantSpaceBP::DEFAULT_ITERS;  // 8
	const int numLevels = cv::gpu::StereoConstantSpaceBP::DEFAULT_LEVELS;  // 4
	const int nrPlanes = cv::gpu::StereoConstantSpaceBP::DEFAULT_NR_PLANE;  // 4
	const int msgType = CV_32F;
    cv::gpu::StereoConstantSpaceBP csbp(numDisparities, numIterations, numLevels, nrPlanes, msgType);

	const int64 start = cv::getTickCount();

	csbp(buffer.imgL, buffer.imgR, buffer.imgDisp, stream);

	const int64 elapsed = cv::getTickCount() - start;
    const double freq = cv::getTickFrequency();
	const double etime = elapsed * 1000.0 / freq;
    const double fps = freq / elapsed;
	std::cout << std::setprecision(4) << "[CSBP] elapsed time: " << etime <<  ", FPS: " << fps << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void stereo_matching_using_gpu()
{
	// [ref] {OPENCV_HOME}/samples/gpu/stereo_match.cpp

	const std::string imageL_filename("./machine_vision_data/tsukuba-imL.png");
    const std::string imageR_filename("./machine_vision_data/tsukuba-imR.png");
    const std::string imageT_filename("./machine_vision_data/tsukuba-truedispL.png");
	const std::size_t NUM_DISPARITIES = 24;  // 8x

	const cv::Mat &imgL = cv::imread(imageL_filename, CV_LOAD_IMAGE_COLOR);
	const cv::Mat &imgR = cv::imread(imageR_filename, CV_LOAD_IMAGE_COLOR);
	const cv::Mat &imgT = cv::imread(imageT_filename, CV_LOAD_IMAGE_COLOR);
	if (imgL.empty() || imgR.empty())
	{
		std::cout << "image files not found ..." << std::endl;
		return;
	}

    cv::imshow("left", imgL);
    cv::imshow("right", imgR);

	//
	local::BufferForStereoMatching buffer;

	//cv::Mat imgDisp(imgL.size(), CV_8U);
	//buffer.imgDisp.create(imgL.size(), CV_8U);
	cv::Mat imgDisp;

	cv::gpu::Stream stream;
	if (true)
	{
		// BM doesn't support color images.

		cv::Mat grayL, grayR;

        cv::cvtColor(imgL, grayL, CV_BGR2GRAY);
        cv::cvtColor(imgR, grayR, CV_BGR2GRAY);

		buffer.imgL.upload(grayL);
		buffer.imgR.upload(grayR);

		//local::bm(NUM_DISPARITIES, buffer);
		local::bm(NUM_DISPARITIES, buffer, stream);

		//
		buffer.imgDisp.download(imgDisp);
		imgDisp.convertTo(imgDisp, CV_8U, 255.0 / (double)NUM_DISPARITIES, 0.0);
		cv::imshow("disparity map - BM", imgDisp);
	}

	if (true)
	{
		buffer.imgL.upload(imgL);
		buffer.imgR.upload(imgR);

		//local::belief_propagation(NUM_DISPARITIES, buffer);
		local::belief_propagation(NUM_DISPARITIES, buffer, stream);

		//
		buffer.imgDisp.download(imgDisp);
		imgDisp.convertTo(imgDisp, CV_8U, 255.0 / (double)NUM_DISPARITIES, 0.0);
		cv::imshow("disparity map - BP", imgDisp);
	}

	if (true)
	{
		buffer.imgL.upload(imgL);
		buffer.imgR.upload(imgR);

		//local::constant_space_belief_propagation(NUM_DISPARITIES, buffer);
		local::constant_space_belief_propagation(NUM_DISPARITIES, buffer, stream);

		//
		buffer.imgDisp.download(imgDisp);
		imgDisp.convertTo(imgDisp, CV_8U, 255.0 / (double)NUM_DISPARITIES, 0.0);
		cv::imshow("disparity map - CSBP", imgDisp);
	}
	cv::waitKey();

	cv::destroyAllWindows();
}

}  // namespace my_opencv
