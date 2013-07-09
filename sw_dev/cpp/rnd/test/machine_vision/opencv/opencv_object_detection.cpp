//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <algorithm>
#include <list>
#include <vector>
#include <string>

#ifdef WIN32
#include <io.h>
#else
#include <dirent.h>
#endif

#ifdef HAVE_CVCONFIG_H
#include <cvconfig.h>
#endif

#ifdef HAVE_TBB
#include <tbb/task_scheduler_init.h>
#endif


namespace {
namespace local {

void detect_and_draw_objects(IplImage *image, CvLatentSvmDetector *detector, const int numThreads)
{
#ifdef HAVE_TBB
	tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
	if (numThreads > 0)
	{
		init.initialize(numThreads);
		std::cout << "Number of threads " << numThreads << std::endl;
	}
	else
	{
		std::cout << "Number of threads is not correct for TBB version" << std::endl;
		return;
	}
#endif

	CvMemStorage *storage = cvCreateMemStorage(0);

	const int64 startint64 = cvGetTickCount();
	CvSeq *detections = cvLatentSvmDetectObjects(image, detector, storage, 0.5f, numThreads);
	const int64 finishint64 = cvGetTickCount();
	printf("detection time = %.3f\n", (float)(finishint64 - startint64) / (float)(cvGetTickFrequency() * 1000000.0));

#ifdef HAVE_TBB
	init.terminate();
#endif

	for (int i = 0; i < detections->total; ++i)
	{
		CvObjectDetection detection = *(CvObjectDetection *)cvGetSeqElem(detections, i);
		CvRect bounding_box = detection.rect;
		cvRectangle(
			image, cvPoint(bounding_box.x, bounding_box.y),
			cvPoint(bounding_box.x + bounding_box.width, bounding_box.y + bounding_box.height),
			CV_RGB(255, 0, 0), 3
		);
	}

	cvReleaseMemStorage(&storage);
}

// [ref]
//	${OPENCV_HOME}/samples/c/latentsvmdetect.cpp
//	"Object Detection with Discriminatively Trained Part Based Models", P. Felzenszwalb, R. Girshick, D. McAllester, & D. Ramanan, TPAMI, 2010.
void latentsvmdetect_sample()
{
	const std::string model_filename("./machine_vision_data/opencv/object_detection/models_VOC2007/cat.xml");
	const std::string image_filename("./machine_vision_data/opencv/object_detection/cat.png");
	const int tbbNumThreads = 4;

	IplImage *image = cvLoadImage(image_filename.c_str());
	if (!image)
	{
		std::cerr << "image file not found: " << image_filename << std::endl;
		return;
	}
	CvLatentSvmDetector *detector = cvLoadLatentSvmDetector(model_filename.c_str());
	if (!detector)
	{
		std::cerr << "model file not found: " << model_filename << std::endl;
		cvReleaseImage(&image);
		return;
	}

	detect_and_draw_objects(image, detector, tbbNumThreads);
	cvNamedWindow("test", 0);
	cvShowImage("test", image);
	cvWaitKey(0);

	cvDestroyAllWindows();

	cvReleaseLatentSvmDetector(&detector);
	cvReleaseImage(&image);
}

void detect_and_draw_objects(cv::Mat &image, cv::LatentSvmDetector &detector, const std::vector<cv::Scalar> &colors, const float overlapThreshold, const int numThreads)
{
    std::vector<cv::LatentSvmDetector::ObjectDetection> detections;

    cv::TickMeter tm;
    tm.start();
    detector.detect(image, detections, overlapThreshold, numThreads);
    tm.stop();

    std::cout << "Detection time = " << tm.getTimeSec() << " sec" << std::endl;

    const std::vector<std::string> classNames = detector.getClassNames();
    CV_Assert(colors.size() == classNames.size());

    for (std::size_t i = 0; i < detections.size(); ++i)
    {
        const cv::LatentSvmDetector::ObjectDetection &od = detections[i];
        cv::rectangle(image, od.rect, colors[od.classID], 3);
    }

    // put text over the all rectangles
    for (std::size_t i = 0; i < detections.size(); ++i)
    {
        const cv::LatentSvmDetector::ObjectDetection &od = detections[i];
        cv::putText(image, classNames[od.classID], cv::Point(od.rect.x + 4, od.rect.y + 13), cv::FONT_HERSHEY_SIMPLEX, 0.55, colors[od.classID], 2);
    }
}

void read_directory(const std::string &directoryName, std::vector<std::string> &filenames, const bool addDirectoryName)
{
    filenames.clear();

#if 1
	if (!boost::filesystem::exists(directoryName))
	{
		std::cerr << "directory not found: " << directoryName << std::endl;
		return;
	}

	boost::filesystem::directory_iterator end_itr;  // default construction yields past-the-end
	for (boost::filesystem::directory_iterator itr(directoryName); itr != end_itr; ++itr)
	{
		if (boost::filesystem::is_regular_file(itr->status()))
		{
            if (addDirectoryName)
				filenames.push_back(directoryName + "/" + itr->path().filename().string());
            else
                filenames.push_back(itr->path().filename().string());
		}
	}
#elif defined(WIN32)
    struct _finddata_t s_file;
    std::string str = directoryName + "\\*.*";

    intptr_t h_file = _findfirst(str.c_str(), &s_file);
    if (h_file != static_cast<intptr_t>(-1.0))
    {
        do
        {
            if (addDirectoryName)
                filenames.push_back(directoryName + "\\" + s_file.name);
            else
                filenames.push_back((std::string)s_file.name);
        } while (_findnext(h_file, &s_file) == 0);
    }
    _findclose(h_file);
#else
    DIR *dir = opendir(directoryName.c_str());
    if (NULL !=  dir)
    {
        struct dirent *dent;
        while (NULL != (dent = readdir(dir)))
        {
            if (addDirectoryName)
                filenames.push_back(directoryName + "/" + std::string(dent->d_name));
            else
                filenames.push_back(std::string(dent->d_name));
        }
    }
#endif

    std::sort(filenames.begin(), filenames.end());
}

// [ref]
//	${OPENCV_HOME}/samples/cpp/latentsvm_multidetect.cpp
//	"Object Detection with Discriminatively Trained Part Based Models", P. Felzenszwalb, R. Girshick, D. McAllester, & D. Ramanan, TPAMI, 2010.
void latentsvm_multidetect_sample()
{
	const std::string images_directory("./machine_vision_data/opencv/object_detection");
	// [ref] ${OPENCV_EXTRA_HOME}/testdata/cv/latentsvmdetector/models_VOC2007
	const std::string models_directory("./machine_vision_data/opencv/object_detection/models_VOC2007");
	const float overlapThreshold = 0.2f;  // [0, 1]
	const int numThreads = 4;

	std::vector<std::string> images_filenames, models_filenames;
	read_directory(images_directory, images_filenames, true);
	read_directory(models_directory, models_filenames, true);

	cv::LatentSvmDetector detector(models_filenames);
	if (detector.empty())
	{
		std::cout << "Models cannot be loaded" << std::endl;
		return;
	}

	const std::vector<std::string> &classNames = detector.getClassNames();
	std::cout << "Loaded " << classNames.size() << " models:" << std::endl;
	for (std::size_t i = 0; i < classNames.size(); ++i)
		std::cout << i << ") " << classNames[i] << std::endl;
	std::cout << std::endl;

	std::cout << "overlapThreshold = " << overlapThreshold << std::endl;

	std::vector<cv::Scalar> colors;
	cv::generateColors(colors, detector.getClassNames().size());

	for (std::size_t i = 0; i < images_filenames.size(); ++i)
	{
		cv::Mat image = cv::imread(images_filenames[i]);
		if (image.empty()) continue;

		std::cout << "Process image " << images_filenames[i] << std::endl;
		detect_and_draw_objects(image, detector, colors, overlapThreshold, numThreads);

		cv::imshow("result", image);

		const int c = cv::waitKey(0);
		if ((char)c == 'n')  // go to the next image
			break;
		else if ((char)c == '\x1b') // quit
			return;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void object_detection()
{
	local::latentsvmdetect_sample();
	local::latentsvm_multidetect_sample();
}

}  // namespace my_opencv
