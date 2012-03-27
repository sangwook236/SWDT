//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <string>
#include <iostream>
#include <stdexcept>


void print_opencv_matrix(const CvMat* mat);

void convert_bmp_to_pgm();
void convert_pgm_to_png();
void convert_ppm_to_png();

void convert_image(const std::string &srcImageName, const std::string &dstImageName);
void convert_image_to_gray(const std::string &srcImageName, const std::string &dstImageName);
void convert_image(const std::string &imageName, const std::string &srcImageExt, const std::string &dstImageExt);
void convert_images(const std::string &dirName, const std::string &srcImageExt, const std::string &dstImageExt);

void image_conversion()
{
	//convert_bmp_to_pgm();
	//convert_pgm_to_png();
	convert_ppm_to_png();
}

void convert_image(const std::string &srcImageName, const std::string &dstImageName)
{
	const IplImage *srcImage = cvLoadImage(srcImageName.c_str());
	if (srcImage)
		cvSaveImage(dstImageName.c_str(), srcImage);
}

void convert_image_to_gray(const std::string &srcImageName, const std::string &dstImageName)
{
	const IplImage *srcImage = cvLoadImage(srcImageName.c_str());
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
#else
			if (_stricmp(srcImage->channelSeq, "RGB") == 0)
#endif
				cvCvtColor(srcImage, grayImg, CV_RGB2GRAY);
#if defined(__GNUC__)
			else if (strcasecmp(srcImage->channelSeq, "BGR") == 0)
#else
			else if (_stricmp(srcImage->channelSeq, "BGR") == 0)
#endif
				cvCvtColor(srcImage, grayImg, CV_BGR2GRAY);
			else
				assert(false);
		}

		if (grayImg)
			cvSaveImage(dstImageName.c_str(), grayImg);
	}
}

void convert_image(const std::string &imageName, const std::string &srcImageExt, const std::string &dstImageExt)
{
	const std::string::size_type extPos = imageName.find_last_of('.');

#if defined(__GNUC__)
	if (strcasecmp(imageName.substr(extPos + 1).c_str(), srcImageExt.c_str()) == 0)
#else
	if (_stricmp(imageName.substr(extPos + 1).c_str(), srcImageExt.c_str()) == 0)
#endif
	{
		const std::string repstr = std::string(imageName).replace(extPos + 1, imageName.length() - extPos -1, dstImageExt);
		//convert_image(imageName, repstr);
		convert_image_to_gray(imageName, repstr);
	}
}

void convert_images(const std::string &dirName, const std::string &srcImageExt, const std::string &dstImageExt)
{
#if defined(WIN32)
	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError;

	hFind = FindFirstFileA((dirName + std::string("\\*")).c_str(), &FindFileData);
	if (INVALID_HANDLE_VALUE == hFind)
	{
		std::cout << "Invalid file handle. Error is " << GetLastError() << std::endl;
		return;
	}
	else
	{
		convert_image(dirName + std::string("\\") + std::string(FindFileData.cFileName), srcImageExt, dstImageExt);

		while (FindNextFileA(hFind, &FindFileData) != 0)
		{
			convert_image(dirName + std::string("\\") + std::string(FindFileData.cFileName), srcImageExt, dstImageExt);
		}

		dwError = GetLastError();
		FindClose(hFind);
		if (ERROR_NO_MORE_FILES != dwError)
		{
			std::cout << "FindNextFile error. Error is " << dwError << std::endl;
			return;
		}
	}
#else
    throw std::runtime_error("not yet implemented");
#endif
}

void convert_bmp_to_pgm()
{
	void convert_images(const std::string &dirName, const std::string &srcImageExt, const std::string &dstImageExt);

	const std::string dirName(".");
	convert_images(dirName, "bmp", "pgm");
}

void convert_pgm_to_png()
{
	void convert_images(const std::string &dirName, const std::string &srcImageExt, const std::string &dstImageExt);

	const std::string dirName(".");
	convert_images(dirName, "pgm", "png");
}

void convert_ppm_to_png()
{
	void convert_images(const std::string &dirName, const std::string &srcImageExt, const std::string &dstImageExt);

	const std::string dirName(".");
	convert_images(dirName, "ppm", "png");
}
