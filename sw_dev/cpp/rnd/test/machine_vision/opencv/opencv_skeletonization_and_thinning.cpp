//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <string>


namespace {
namespace local {

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * Parameters:
 * 		im    Binary image with range = [0,1]
 * 		iter  0=even, 1=odd
 */
void thinningZhangSuenIteration(cv::Mat& img, int iter)
{
	CV_Assert(1 == img.channels());
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	uchar *nw, *no, *ne;    // north (pAbove)
	uchar *we, *me, *ea;
	uchar *sw, *so, *se;    // south (pBelow)

	uchar *pDst;

	// initialize row pointers
	uchar *pAbove = NULL;
	uchar *pCurr  = img.ptr<uchar>(0);
	uchar *pBelow = img.ptr<uchar>(1);

	int x, y;
	for (y = 1; y < img.rows - 1; ++y)
	{
		// shift the rows up by one
		pAbove = pCurr;
		pCurr  = pBelow;
		pBelow = img.ptr<uchar>(y+1);

		pDst = marker.ptr<uchar>(y);

		// initialize col pointers
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x)
		{
			// shift col pointers left by one (scan left to right)
			nw = no;
			no = ne;
			ne = &(pAbove[x+1]);
			we = me;
			me = ea;
			ea = &(pCurr[x+1]);
			sw = so;
			so = se;
			se = &(pBelow[x+1]);

			const int A  = (0 == *no && 1 == *ne) + (0 == *ne && 1 == *ea) + 
				(0 == *ea && 1 == *se) + (0 == *se && 1 == *so) + 
				(0 == *so && 1 == *sw) + (0 == *sw && 1 == *we) +
				(0 == *we && 1 == *nw) + (0 == *nw && 1 == *no);
			const int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
			const int m1 = 0 == iter ? (*no * *ea * *so) : (*no * *ea * *we);
			const int m2 = 0 == iter ? (*ea * *so * *we) : (*no * *so * *we);

			if (1 == A && (B >= 2 && B <= 6) && 0 == m1 && 0 == m2)
				pDst[x] = 1;
		}
	}

	img &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * Parameters:
 * 		src  The source image, binary with range = [0,255]
 * 		dst  The destination image
 */
// REF [paper] >> "A fast parallel algorithm for thinning digital patterns", T.Y. Zhang and C.Y. Suen, CACM, 1984.
// REF [site] >>
//	https://github.com/bsdnoobz/zhang-suen-thinning
//	http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/
void zhang_suen_thinning_algorithm(const cv::Mat &src, cv::Mat &dst)
{
	dst = src.clone();
	dst /= 255;  // convert to binary image

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	do
	{
		thinningZhangSuenIteration(dst, 0);
		thinningZhangSuenIteration(dst, 1);
		cv::absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	dst *= 255;
}

/**
* Perform one thinning iteration.
* Normally you wouldn't call this function directly from your code.
*
* @param  im    Binary image with range = 0-1
* @param  iter  0=even, 1=odd
*/
void thinningGuoHallIteration(cv::Mat &im, const int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1); 

	for (int i = 1; i < im.rows; ++i)
	{
		for (int j = 1; j < im.cols; ++j)
		{
			const uchar &p2 = im.at<uchar>(i-1, j);
			const uchar &p3 = im.at<uchar>(i-1, j+1);
			const uchar &p4 = im.at<uchar>(i, j+1);
			const uchar &p5 = im.at<uchar>(i+1, j+1);
			const uchar &p6 = im.at<uchar>(i+1, j);
			const uchar &p7 = im.at<uchar>(i+1, j-1);
			const uchar &p8 = im.at<uchar>(i, j-1); 
			const uchar &p9 = im.at<uchar>(i-1, j-1);

			const int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) + (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
			const int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
			const int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
			const int N  = N1 < N2 ? N1 : N2;
			const int m  = 0 == iter ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

			if (1 == C && (N >= 2 && N <= 3) && 0 == m)
				marker.at<uchar>(i, j) = 1;
		}
	}

	im &= ~marker;
}

/**
* Function for thinning the given binary image
*
* @param  im  Binary image with range = 0-255
*/
// REF [paper] >> "Parallel thinning with two sub-iteration algorithms", Zicheng Guo and Richard Hall, CACM, 1989.
// REF [site] >> http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/
void guo_hall_thinning_algorithm(cv::Mat &im)
{
	im /= 255;

	cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat diff;

	do
	{
		thinningGuoHallIteration(im, 0);
		thinningGuoHallIteration(im, 1);
		cv::absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	im *= 255;
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void skeletonization_and_thinning()
{
	const std::string input_filename("./data/machine_vision/opencv/thinning_img_1.png");
	//const std::string input_filename("./data/machine_vision/opencv/thinning_img_2.jpg");
	const cv::Mat &src = cv::imread(input_filename);
	if (src.empty())
	{
		std::cerr << "file not found: " << input_filename << std::endl;
		return;
	}

	cv::imshow("src image", src);

	// Zhang-Suen thinning algorithm.
	{
		cv::Mat bw;
		cv::cvtColor(src, bw, CV_BGR2GRAY);
		cv::threshold(bw, bw, 10, 255, CV_THRESH_BINARY);

		{
			boost::timer::auto_cpu_timer timer;

			local::zhang_suen_thinning_algorithm(bw, bw);
		}

		cv::imshow("Zhang-Suen thinning algorithm - result", bw);
	}

	// Guo-Hall thinning algorithm.
	{
		cv::Mat gray;
		cv::cvtColor(src, gray, CV_BGR2GRAY);

		{
			boost::timer::auto_cpu_timer timer;

			local::guo_hall_thinning_algorithm(gray);
		}

		cv::imshow("Guo-Hall thinning algorithm - result", gray);
	}

	cv::waitKey();

	cv::destroyAllWindows();
}

}  // namespace my_opencv
