#include <FlyCapture2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace


// REF [site] >> http://www.ptgrey.com/KB/10861.
int flycapture2_main(int argc, char *argv[])
{
	FlyCapture2::Error error;
	FlyCapture2::Camera camera;
	FlyCapture2::CameraInfo camInfo;

	// Connect the camera.
	error = camera.Connect(0);
	if (FlyCapture2::PGRERROR_OK != error.GetType())
	{
		std::cout << "Failed to connect to camera" << std::endl;
		return 1;
	}

	// Get the camera info and print it out.
	error = camera.GetCameraInfo(&camInfo);
	if (FlyCapture2::PGRERROR_OK != error.GetType())
	{
		std::cout << "Failed to get camera info from camera" << std::endl;
		return 1;
	}
	std::cout << camInfo.vendorName << " "
		<< camInfo.modelName << " "
		<< camInfo.serialNumber << std::endl;

	error = camera.StartCapture();
	if (FlyCapture2::PGRERROR_ISOCH_BANDWIDTH_EXCEEDED == error.GetType())
	{
		std::cout << "Bandwidth exceeded" << std::endl;
		return 1;
	}
	else if (FlyCapture2::PGRERROR_OK != error.GetType())
	{
		std::cout << "Failed to start image capture" << std::endl;
		return 1;
	}

	// capture loop.
	char key = 0;
	while (key != 'q')
	{
		// Get the image.
		FlyCapture2::Image rawImage;
		error = camera.RetrieveBuffer(&rawImage);
		if (FlyCapture2::PGRERROR_OK != error.GetType())
		{
			std::cout << "capture error" << std::endl;
			continue;
		}

		// convert to rgb.
		FlyCapture2::Image rgbImage;
		rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);

		// convert to OpenCV Mat.
		unsigned int rowBytes = (double)rgbImage.GetReceivedDataSize() / (double)rgbImage.GetRows();
		cv::Mat image = cv::Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData(), rowBytes);

		cv::imshow("image", image);
		key = cv::waitKey(30);
	}

	error = camera.StopCapture();
	if (FlyCapture2::PGRERROR_OK != error.GetType())
	{
		// This may fail when the camera was removed, so don't show an error message
	}

	camera.Disconnect();

	return 0;
}
