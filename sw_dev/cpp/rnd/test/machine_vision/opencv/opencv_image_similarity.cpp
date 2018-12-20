//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>


namespace {
namespace local {

// Peak signal-to-noise ratio (PSNR) = 10 * log10(max(I)^2 / MSE).
// REF [site] >> https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
// REF [site] >> http://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html
double psnr(const cv::Mat &img1, const cv::Mat &img2)
{
	cv::Mat s1;
	cv::absdiff(img1, img2, s1);   // |img1 - img2|.
	s1.convertTo(s1, CV_32F);  // Cannot make a square on 8 bits.
	s1 = s1.mul(s1);  // |img1 - img2|^2.

	const cv::Scalar s = cv::sum(s1);  // Sum elements per channel.
	const double sse = s.val[0] + s.val[1] + s.val[2];  // Sum channels.

	if (sse <= 1e-10) // For small values return zero.
		return 0.0;
	else
	{
		const double mse = sse / double(img1.channels() * img1.total());
		return 10.0 * std::log10((255.0 * 255.0) / mse);
	}
}

// Mean of structural similarity (SSIM).
// REF [site] >> https://en.wikipedia.org/wiki/Structural_similarity
// REF [site] >> http://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html
cv::Scalar mssim(const cv::Mat &img1, const cv::Mat &img2)
{
	cv::Mat I1, I2;
	img1.convertTo(I1, CV_32F);  // Cannot calculate on one byte large values.
	img2.convertTo(I2, CV_32F);

	const cv::Mat I2_2(I2.mul(I2));  // I2^2.
	const cv::Mat I1_2(I1.mul(I1));  // I1^2.
	const cv::Mat I1_I2(I1.mul(I2));  // I1 * I2.

	// Preliminary computation.
	cv::Mat mu1, mu2;
	cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
	cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

	const cv::Mat mu1_2(mu1.mul(mu1));
	const cv::Mat mu2_2(mu2.mul(mu2));
	const cv::Mat mu1_mu2(mu1.mul(mu2));

	cv::Mat sigma1_2, sigma2_2, sigma12;
	cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	// Formula.
	const double C1 = 6.5025, C2 = 58.5225;
	cv::Mat t1(2 * mu1_mu2 + C1);
	cv::Mat t2(2 * sigma12 + C2);
	cv::Mat t3(t1.mul(t2));  // t3 = ((2*mu1_mu2 + C1) .* (2*sigma12 + C2)).

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);  // t1 = ((mu1_2 + mu2_2 + C1) .* (sigma1_2 + sigma2_2 + C2)).

	cv::Mat ssim_map;
	cv::divide(t3, t1, ssim_map);  // ssim_map = t3 ./ t1.

	return cv::mean(ssim_map);  // mssim = average of ssim map.
}

// REF [file] >> ${OPENCV_HOME}/samples/cpp/tutorial_code/videoio/video-input-psnr-ssim/video-input-psnr-ssim.cpp
// REF [site] >> http://docs.opencv.org/3.2/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html#image-similarity-psnr-and-ssim
void video_similarity()
{
	const std::string sourceReference("../data/machine_vision/Megamind.avi"), sourceCompareWith("./data/machine_vision/Megamind_bugy.avi");
	const int psnrTriggerValue = 35, delay = 10;

	int frameNum = -1;  // Frame counter.

	cv::VideoCapture captRefrnc(sourceReference), captUndTst(sourceCompareWith);
	if (!captRefrnc.isOpened())
	{
		std::cout << "Could not open reference " << sourceReference << std::endl;
		return;
	}
	if (!captUndTst.isOpened())
	{
		std::cout << "Could not open case test " << sourceCompareWith << std::endl;
		return;
	}

	const cv::Size refS = cv::Size((int)captRefrnc.get(cv::CAP_PROP_FRAME_WIDTH), (int)captRefrnc.get(cv::CAP_PROP_FRAME_HEIGHT)),
		uTSi = cv::Size((int)captUndTst.get(cv::CAP_PROP_FRAME_WIDTH), (int)captUndTst.get(cv::CAP_PROP_FRAME_HEIGHT));
	if (refS != uTSi)
	{
		std::cout << "Inputs have different size!!! Closing." << std::endl;
		return;
	}

	const std::string WIN_UT("Under Test");
	const std::string WIN_RF("Reference");

	// Windows.
	cv::namedWindow(WIN_RF, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(WIN_UT, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(WIN_RF, 400, 0);  //750, 2 (bernat = 0).
	cv::moveWindow(WIN_UT, refS.width, 0);  //1500, 2.

	std::cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height
		<< " of nr#: " << captRefrnc.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;

	std::cout << "PSNR trigger value " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << psnrTriggerValue << std::endl;

	cv::Mat frameReference, frameUnderTest;
	double psnrV;
	cv::Scalar mssimV;

	for (;;) // Show the image captured in the window and repeat.
	{
		captRefrnc >> frameReference;
		captUndTst >> frameUnderTest;

		if (frameReference.empty() || frameUnderTest.empty())
		{
			std::cout << " < < <  Game over!  > > > ";
			break;
		}

		++frameNum;
		std::cout << "Frame: " << frameNum << "# ";

		// Peak signal-to-noise ratio (PSNR).
		psnrV = psnr(frameReference, frameUnderTest);
		std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << psnrV << "dB";

		// Mean of structural similarity (MSSIM).
		if (psnrV < psnrTriggerValue && psnrV)
		{
			mssimV = mssim(frameReference, frameUnderTest);

			std::cout << " MSSIM: "
				<< " R " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << mssimV.val[2] * 100 << "%"
				<< " G " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << mssimV.val[1] * 100 << "%"
				<< " B " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << mssimV.val[0] * 100 << "%";
		}
		std::cout << std::endl;

		// Show image.
		cv::imshow(WIN_RF, frameReference);
		cv::imshow(WIN_UT, frameUnderTest);

		char c = (char)cv::waitKey(delay);
		if (c == 27) break;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_similarity()
{
	local::video_similarity();
}

}  // namespace my_opencv
