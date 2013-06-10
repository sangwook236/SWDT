//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <boost/filesystem.hpp>
#include <string>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void convert_image(const std::string &srcImageName, const std::string &dstImageName)
{
#if 0
	IplImage *srcImage = cvLoadImage(srcImageName.c_str());
	if (srcImage)
	{
		cvSaveImage(dstImageName.c_str(), srcImage);
		cvReleaseImage(&srcImage);
	}
#else
	cv::Mat src = cv::imread(srcImageName);
	if (!src.empty())
	{
		cv::imwrite(dstImageName, src);
		std::cout << "succeed in converting an image file, " << srcImageName << " to another image file, " << dstImageName << std::endl;
	}
	else
		std::cerr << "fail to read an image file, " << srcImageName << std::endl;
#endif
}

void convert_image_to_gray(const std::string &srcImageName, const std::string &dstImageName)
{
#if 0
	IplImage *srcImage = cvLoadImage(srcImageName.c_str());
	if (srcImage)
	{
		IplImage *grayImg = 0L;
		if (1 == srcImage->nChannels)
			grayImg = const_cast<IplImage *>(srcImage);
		else
		{
			grayImg = cvCreateImage(cvGetSize(srcImage), srcImage->depth, 1);
#if defined(__GNUC__)
			if (strcasecmp(srcImage->channelSeq, "RGB") == 0)
#elif defined(_MSC_VER)
			if (_stricmp(srcImage->channelSeq, "RGB") == 0)
#endif
				cvCvtColor(srcImage, grayImg, CV_RGB2GRAY);
#if defined(__GNUC__)
			else if (strcasecmp(srcImage->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
			else if (_stricmp(srcImage->channelSeq, "BGR") == 0)
#endif
				cvCvtColor(srcImage, grayImg, CV_BGR2GRAY);
			else
				assert(false);
		}

		if (grayImg)
			cvSaveImage(dstImageName.c_str(), grayImg);

		if (grayImg != srcImage)
			cvReleaseImage(&grayImg);
		cvReleaseImage(&srcImage);
	}
#else
	cv::Mat srcImage = cv::imread(srcImageName);
	if (!srcImage.empty())
	{
		if (1 == srcImage.channels())
			cv::imwrite(dstImageName, srcImage);
		else
		{
			cv::Mat grayImg;
			// FIXME [enhance] >>
			//cv::cvtColor(srcImage, grayImg, CV_RGB2GRAY);
			cv::cvtColor(srcImage, grayImg, CV_BGR2GRAY);

			if (!grayImg.empty())
				cv::imwrite(dstImageName, grayImg);
		}

		std::cout << "succeed in converting an image file, " << srcImageName << " to a gray image file, " << dstImageName << std::endl;
	}
	else
		std::cerr << "fail to read an image file, " << srcImageName << std::endl;
#endif
}

void convert_image(const std::string &imageName, const std::string &srcImageExt, const std::string &dstImageExt, const bool isGray)
{
	const std::string::size_type extPos = imageName.find_last_of('.');

#if defined(__GNUC__)
	if (strcasecmp(imageName.substr(extPos + 1).c_str(), srcImageExt.c_str()) == 0)
#elif defined(_MSC_VER)
	if (_stricmp(imageName.substr(extPos + 1).c_str(), srcImageExt.c_str()) == 0)
#endif
	{
		const std::string repstr = std::string(imageName).replace(extPos + 1, imageName.length() - extPos -1, dstImageExt);

		if (isGray)
			convert_image_to_gray(imageName, repstr);
		else
			convert_image(imageName, repstr);
	}
}

void convert_images(const std::string &dirName, const std::string &srcImageExt, const std::string &dstImageExt, const bool isGray)
{
#if defined(WIN32) && 0
	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError;

	hFind = FindFirstFileA((dirName + std::string("/*")).c_str(), &FindFileData);
	if (INVALID_HANDLE_VALUE == hFind)
	{
		std::cout << "Invalid file handle. Error is " << GetLastError() << std::endl;
		return;
	}
	else
	{
		do
		{
			convert_image(dirName + '/' + std::string(FindFileData.cFileName), srcImageExt, dstImageExt);
		} while (FindNextFileA(hFind, &FindFileData) != 0);

		dwError = GetLastError();
		FindClose(hFind);
		if (ERROR_NO_MORE_FILES != dwError)
		{
			std::cout << "FindNextFile error. Error is " << dwError << std::endl;
			return;
		}
	}
	
	CloseHandle(hFind);
#else
	const boost::filesystem::path dir_path(dirName);

	if (!boost::filesystem::exists(dir_path))
	{
		std::cout << "incorrect path: the assigned path is not a directory" << std::endl;
		return;
	}

	boost::filesystem::directory_iterator end_itr;  // default construction yields past-the-end
	for (boost::filesystem::directory_iterator itr(dir_path); itr != end_itr; ++itr)
	{
		convert_image(dirName + '/' + itr->path().filename().string(), srcImageExt, dstImageExt, isGray);
	}
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_conversion()
{
	//const std::string dir_name(".");
	//const std::string dir_name("E:/archive_dataset/change_detection/canoe/input");
	//const std::string dir_name("E:/archive_dataset/change_detection/highway/input");
	//const std::string dir_name("E:/archive_dataset/change_detection/boats/input");
	const std::string dir_name("D:/work_center/sw_dev/cpp/rnd/bin/segmentation_data");

#if 0
	const std::string src_ext("bmp");
	const std::string dst_ext("pgm");
#elif 0
	const std::string src_ext("pgm");
	const std::string dst_ext("png");
#elif 0
	const std::string src_ext("ppm");
	const std::string dst_ext("png");
#elif 0
	const std::string src_ext("jpg");
	const std::string dst_ext("ppm");
#elif 1
	const std::string src_ext("png");
	const std::string dst_ext("ppm");
#endif

	const bool is_gray = false;
	local::convert_images(dir_name, src_ext, dst_ext, is_gray);
}

}  // namespace my_opencv
