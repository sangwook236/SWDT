#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <list>
#include <ctime>


namespace {
namespace local {

//-----------------------------------------------------------------------------
// [ref] "Human Skin Colour Clustering for Face Detection"
// Peter Peer, Jure Kovac, & Franc Solina, Eurocon 2003
struct PeerSkinColorModel
{
	static void filter(const cv::Mat &img, const bool is_uniform_daylight, cv::Mat &mask);
};

/*static*/ void PeerSkinColorModel::filter(const cv::Mat &img, const bool is_uniform_daylight, cv::Mat &mask)
{
	mask = cv::Mat::zeros(img.size(), CV_8U);

	if (is_uniform_daylight)
	{
		// the skin colour at uniform daylight illumination
		// R > 95 && G > 40 && B > 20 &&
		// max(R, G, B) - min(R, G, B) > 15 &&
		// abs(R - G) > 15 &&
		// R > G && R > B

		const unsigned char *imgPtr = img.data;
		unsigned char *maskPtr = mask.data;
		for (int r = 0; r < img.rows; ++r)
		{
			for (int c = 0; c < img.cols; ++c, imgPtr += 3, ++maskPtr)
			{
				const unsigned char &red = imgPtr[2];
				const unsigned char &green = imgPtr[1];
				const unsigned char &blue = imgPtr[0];

				if (red > 95 && green > 40 && blue > 20 && (std::max(std::max(red, green), blue) - std::min(std::min(red, green), blue)) > 15 && std::abs(red - green) > 15 && red > green && red > blue)
					*maskPtr = 1;
			}
		}
	}
	else
	{
		// the skin colour under flashlight or (light) daylight lateral illumination
		// R > 220 && G > 210 && B > 170 &&
		// abs(R - G) > 15 &&
		// R > G && R > B

		const unsigned char *imgPtr = img.data;
		unsigned char *maskPtr = mask.data;
		for (int r = 0; r < img.rows; ++r)
		{
			for (int c = 0; c < img.cols; ++c, imgPtr += 3, ++maskPtr)
			{
				const unsigned char &red = imgPtr[2];
				const unsigned char &green = imgPtr[1];
				const unsigned char &blue = imgPtr[0];

				if (red > 220 && green > 210 && blue > 170 && std::abs(red - green) > 15 && red > green && red > blue)
					*maskPtr = 1;
			}
		}
	}
}

//-----------------------------------------------------------------------------
// [ref] "Statistical Color Models with Application to Skin Color"
// Michael J. Jones & James M. Rehg, IJCV 2002
class GmmSkinColorModel
{
public:
	static const size_t GAUSSIAN_COMPONENT_NUM = 16;
	static const size_t COLOR_CHANNEL_NUM = 16;

private:
	GmmSkinColorModel();

public:
	static GmmSkinColorModel & getInstance();

public:
	void filter(const cv::Mat &img, const float theta, cv::Mat &mask) const;

	const cv::Mat & getMuSkin() const  {  return mu_skin_;  }
	const cv::Mat & getSigmaSkin() const  {  return sigma_skin_;  }
	const cv::Mat & getMuNonskin() const  {  return mu_nonskin_;  }
	const cv::Mat & getSigmaNonskin() const  {  return sigma_nonskin_;  }

	const cv::Mat & getConstSkin() const  {  return const_skin_;  }
	const cv::Mat & getConstNonskin() const  {  return const_nonskin_;  }

private:
	// mixture of Gaussian skin color model
	static const float mu_skin_arr[GAUSSIAN_COMPONENT_NUM * COLOR_CHANNEL_NUM];
	static const float sigma_skin_arr[GAUSSIAN_COMPONENT_NUM * COLOR_CHANNEL_NUM];
	static const float weight_skin_arr[GAUSSIAN_COMPONENT_NUM];
	// mixture of Gaussian non-skin color model
	static const float mu_nonskin_arr[GAUSSIAN_COMPONENT_NUM * COLOR_CHANNEL_NUM];
	static const float sigma_nonskin_arr[GAUSSIAN_COMPONENT_NUM * COLOR_CHANNEL_NUM];
	static const float weight_nonskin_arr[GAUSSIAN_COMPONENT_NUM];

private:
	const cv::Mat mu_skin_;
	const cv::Mat sigma_skin_;
	const cv::Mat mu_nonskin_;
	const cv::Mat sigma_nonskin_;

	cv::Mat const_skin_;
	cv::Mat const_nonskin_;
};

/*static*/ const float GmmSkinColorModel::mu_skin_arr[] = {
	73.53f, 29.94f, 17.76f,
	249.71f, 233.94f, 217.49f,
	161.68f, 116.25f, 96.95f,
	186.07f, 136.62f, 114.40f,
	189.26f, 98.37f, 51.18f,
	247.00f, 152.20f, 90.84f,
	150.10f, 72.66f, 37.76f,
	206.85f, 171.09f, 156.34f,
	212.78f, 152.82f, 120.04f,
	234.87f, 175.43f, 138.94f,
	151.19f, 97.74f, 74.59f,
	120.52f, 77.55f, 59.82f,
	192.20f, 119.62f, 82.32f,
	214.29f, 136.08f, 87.24f,
	99.57f, 54.33f, 38.06f,
	238.88f, 203.08f, 176.91f
};

/*static*/ const float GmmSkinColorModel::sigma_skin_arr[] = {
	765.40f, 121.44f, 112.80f,
	39.94f, 154.44f, 396.05f,
	291.03f, 60.48f, 162.85f,
	274.95f, 64.60f, 198.27f,
	633.18f, 222.40f, 250.69f,
	65.23f, 691.53f, 609.92f,
	408.63f, 200.77f, 257.57f,
	530.08f, 155.08f, 572.79f,
	160.57f, 84.52f, 243.90f,
	163.80f, 121.57f, 279.22f,
	425.40f, 73.56f, 175.11f,
	330.45f, 70.34f, 151.82f,
	152.76f, 92.14f, 259.15f,
	204.90f, 140.17f, 270.19f,
	448.13f, 90.18f, 151.29f,
	178.38f, 156.27f, 404.99f
};

/*static*/ const float GmmSkinColorModel::weight_skin_arr[] = {
	0.0294f,
	0.0331f,
	0.0654f,
	0.0756f,
	0.0554f,
	0.0314f,
	0.0454f,
	0.0469f,
	0.0956f,
	0.0763f,
	0.1100f,
	0.0676f,
	0.0755f,
	0.0500f,
	0.0667f,
	0.0749f
};

/*static*/ const float GmmSkinColorModel::mu_nonskin_arr[] = {
	254.37f, 254.41f, 253.82f,
	9.39f, 8.09f, 8.52f,
	96.57f, 96.95f, 91.53f,
	160.44f, 162.49f, 159.06f,
	74.98f, 63.23f, 46.33f,
	121.83f, 60.88f, 18.31f,
	202.18f, 154.88f, 91.04f,
	193.06f, 201.93f, 206.55f,
	51.88f, 57.14f, 61.55f,
	30.88f, 26.84f, 25.32f,
	44.97f, 85.96f, 131.95f,
	236.02f, 236.27f, 230.70f,
	207.86f, 191.20f, 164.12f,
	99.83f, 148.11f, 188.17f,
	135.06f, 131.92f, 123.10f,
	135.96f, 103.89f, 66.88f
};

/*static*/ const float GmmSkinColorModel::sigma_nonskin_arr[] = {
	2.77f, 2.81f, 5.46f,
	46.84f, 33.59f, 32.48f,
	280.69f, 156.79f, 436.58f,
	355.98f, 115.89f, 591.24f,
	414.84f, 245.95f, 361.27f,
	2502.24f, 1383.53f, 237.18f,
	957.42f, 1766.94f, 1582.52f,
	562.88f, 190.23f, 447.28f,
	344.11f, 191.77f, 433.40f,
	222.07f, 118.65f, 182.41f,
	651.32f, 840.52f, 963.67f,
	225.03f, 117.29f, 331.95f,
	494.04f, 237.69f, 533.52f,
	955.88f, 654.95f, 916.70f,
	350.35f, 130.30f, 388.43f,
	806.44f, 642.20f, 350.36f
};

/*static*/ const float GmmSkinColorModel::weight_nonskin_arr[] = {
	0.0637f,
	0.0516f,
	0.0864f,
	0.0636f,
	0.0747f,
	0.0365f,
	0.0349f,
	0.0649f,
	0.0656f,
	0.1189f,
	0.0362f,
	0.0849f,
	0.0368f,
	0.0389f,
	0.0943f,
	0.0477f
};

/*static*/ GmmSkinColorModel & GmmSkinColorModel::getInstance()
{
	static GmmSkinColorModel aGmmSkinColorModel;
	return aGmmSkinColorModel;
}

GmmSkinColorModel::GmmSkinColorModel()
: mu_skin_(cv::Mat(GAUSSIAN_COMPONENT_NUM, 3, CV_32F, (void *)GmmSkinColorModel::mu_skin_arr, cv::Mat::AUTO_STEP)),
  sigma_skin_(cv::Mat(GAUSSIAN_COMPONENT_NUM, 3, CV_32F, (void *)GmmSkinColorModel::sigma_skin_arr, cv::Mat::AUTO_STEP)),
  mu_nonskin_(cv::Mat(GAUSSIAN_COMPONENT_NUM, 3, CV_32F, (void *)GmmSkinColorModel::mu_nonskin_arr, cv::Mat::AUTO_STEP)),
  sigma_nonskin_(cv::Mat(GAUSSIAN_COMPONENT_NUM, 3, CV_32F, (void *)GmmSkinColorModel::sigma_nonskin_arr, cv::Mat::AUTO_STEP))
{
	const cv::Mat weight_skin(GAUSSIAN_COMPONENT_NUM, 1, CV_32F, (void *)GmmSkinColorModel::weight_skin_arr, cv::Mat::AUTO_STEP);
	const cv::Mat weight_nonskin(GAUSSIAN_COMPONENT_NUM, 1, CV_32F, (void *)GmmSkinColorModel::weight_nonskin_arr, cv::Mat::AUTO_STEP);

	cv::Mat mat_tmp;
	cv::sqrt(sigma_skin_(cv::Range::all(), cv::Range(0,1)).mul(sigma_skin_(cv::Range::all(), cv::Range(1,2))).mul(sigma_skin_(cv::Range::all(), cv::Range(2,3)), 8*CV_PI*CV_PI*CV_PI), mat_tmp);
	const_skin_ = weight_skin / mat_tmp;
	cv::sqrt(sigma_nonskin_(cv::Range::all(), cv::Range(0,1)).mul(sigma_nonskin_(cv::Range::all(), cv::Range(1,2))).mul(sigma_nonskin_(cv::Range::all(), cv::Range(2,3)), 8*CV_PI*CV_PI*CV_PI), mat_tmp);
	const_nonskin_ = weight_nonskin / mat_tmp;
};

void GmmSkinColorModel::filter(const cv::Mat &img, const float theta, cv::Mat &mask) const
{
	mask = cv::Mat::zeros(img.size(), CV_8U);

	const unsigned char *imgPtr = img.data;
	unsigned char *maskPtr = mask.data;
#if 0
	float rgb[3] = { 0.0f, };
	cv::Mat mat_tmp, vec_tmp;
	for (int r = 0; r < img.rows; ++r)
	{
		for (int c = 0; c < img.cols; ++c, imgPtr += 3, ++maskPtr)
		{
			rgb[0] = (float)imgPtr[2];
			rgb[1] = (float)imgPtr[1];
			rgb[2] = (float)imgPtr[0];

			const cv::Mat rgb_mat(cv::repeat(cv::Mat(1, 3, CV_32F, (void *)rgb), GmmSkinColorModel::GAUSSIAN_COMPONENT_NUM, 1));

			// class-conditional pdf of skin color
			cv::pow(rgb_mat - mu_skin_, 2, mat_tmp);
			cv::reduce(mat_tmp / sigma_skin_, vec_tmp, 1, CV_REDUCE_SUM, -1);
			cv::exp(-0.5 * vec_tmp, vec_tmp);

			const cv::Scalar &prob_skin = cv::sum(const_skin_.mul(vec_tmp));

			// class-conditional pdf of nonskin color
			cv::pow(rgb_mat - mu_nonskin_, 2, mat_tmp);
			cv::reduce(mat_tmp / sigma_nonskin_, vec_tmp, 1, CV_REDUCE_SUM, -1);
			cv::exp(-0.5 * vec_tmp, vec_tmp);

			const cv::Scalar &prob_nonskin = cv::sum(const_nonskin_.mul(vec_tmp));

			if (prob_skin[0] >= theta * prob_nonskin[0])
				*maskPtr = 1;
		}
	}
#else
	const float *muSkinPtr, *sigmaSkinPtr, *constSkinPtr, *muNonskinPtr, *sigmaNonskinPtr, *constNonskinPtr;
	float red, green, blue, x0, x1, x2, y0, y1, y2;
	for (int r = 0; r < img.rows; ++r)
	{
		for (int c = 0; c < img.cols; ++c, ++maskPtr)
		{
			muSkinPtr = (float *)mu_skin_.data;
			sigmaSkinPtr = (float *)sigma_skin_.data;
			constSkinPtr = (float *)const_skin_.data;
			muNonskinPtr = (float *)mu_nonskin_.data;
			sigmaNonskinPtr = (float *)sigma_nonskin_.data;
			constNonskinPtr = (float *)const_nonskin_.data;

			blue = (float)*(imgPtr++);
			green = (float)*(imgPtr++);
			red = (float)*(imgPtr++);

			float probSkin = 0.0f, probNonskin = 0.0f;
#if 0
			for (int k = 0; k < GAUSSIAN_COMPONENT_NUM; ++k, muSkinPtr += 3, sigmaSkinPtr += 3, ++constSkinPtr, muNonskinPtr += 3, sigmaNonskinPtr += 3, ++constNonskinPtr)
			{
				x0 = red - muSkinPtr[0];
				x1 = green - muSkinPtr[1];
				x2 = blue - muSkinPtr[2];
				y0 = red - muNonskinPtr[0];
				y1 = green - muNonskinPtr[1];
				y2 = blue - muNonskinPtr[2];

				probSkin += *constSkinPtr * std::exp(-0.5f * (x0*x0/sigmaSkinPtr[0] + x1*x1/sigmaSkinPtr[1] + x2*x2/sigmaSkinPtr[2]));
				probNonskin += *constNonskinPtr * std::exp(-0.5f * (y0*y0/sigmaNonskinPtr[0] + y1*y1/sigmaNonskinPtr[1] + y2*y2/sigmaNonskinPtr[2]));
			}
#else
			for (int k = 0; k < GAUSSIAN_COMPONENT_NUM; ++k)
			{
				x0 = red - *(muSkinPtr++);
				x1 = green - *(muSkinPtr++);
				x2 = blue - *(muSkinPtr++);
				y0 = red - *(muNonskinPtr++);
				y1 = green - *(muNonskinPtr++);
				y2 = blue - *(muNonskinPtr++);

				probSkin += *(constSkinPtr++) * std::exp(-0.5f * (x0*x0 / *(sigmaSkinPtr++) + x1*x1 / *(sigmaSkinPtr++) + x2*x2 / *(sigmaSkinPtr++)));
				probNonskin += *(constNonskinPtr++) * std::exp(-0.5f * (y0*y0 / *(sigmaNonskinPtr++) + y1*y1 / *(sigmaNonskinPtr++) + y2*y2 / *(sigmaNonskinPtr++)));
			}
#endif

			if (probSkin >= theta * probNonskin)
				*maskPtr = 1;
		}
	}
#endif
}


void skin_color_filtering()
{
	std::list<std::string> filenames;
#if 0
	filenames.push_back("opencv_data\\pic1.png");
	filenames.push_back("opencv_data\\pic2.png");
	filenames.push_back("opencv_data\\pic3.png");
	filenames.push_back("opencv_data\\pic4.png");
	filenames.push_back("opencv_data\\pic5.png");
	filenames.push_back("opencv_data\\pic6.png");
	filenames.push_back("opencv_data\\stuff.jpg");
	filenames.push_back("opencv_data\\synthetic_face.png");
	filenames.push_back("opencv_data\\puzzle.png");
	filenames.push_back("opencv_data\\fruits.jpg");
	filenames.push_back("opencv_data\\lena_rgb.bmp");
	filenames.push_back("opencv_data\\hand_01.jpg");
	filenames.push_back("opencv_data\\hand_05.jpg");
	filenames.push_back("opencv_data\\hand_24.jpg");
#elif 1
	filenames.push_back("opencv_data\\hand_left_1.jpg");
	filenames.push_back("opencv_data\\hand_right_1.jpg");

	filenames.push_back("opencv_data\\hand_01.jpg");
	filenames.push_back("opencv_data\\hand_02.jpg");
	filenames.push_back("opencv_data\\hand_03.jpg");
	filenames.push_back("opencv_data\\hand_04.jpg");
	filenames.push_back("opencv_data\\hand_05.jpg");
	filenames.push_back("opencv_data\\hand_06.jpg");
	filenames.push_back("opencv_data\\hand_07.jpg");
	filenames.push_back("opencv_data\\hand_08.jpg");
	filenames.push_back("opencv_data\\hand_09.jpg");
	filenames.push_back("opencv_data\\hand_10.jpg");
	filenames.push_back("opencv_data\\hand_11.jpg");
	filenames.push_back("opencv_data\\hand_12.jpg");
	filenames.push_back("opencv_data\\hand_13.jpg");
	filenames.push_back("opencv_data\\hand_14.jpg");
	filenames.push_back("opencv_data\\hand_15.jpg");
	filenames.push_back("opencv_data\\hand_16.jpg");
	filenames.push_back("opencv_data\\hand_17.jpg");
	filenames.push_back("opencv_data\\hand_18.jpg");
	filenames.push_back("opencv_data\\hand_19.jpg");
	filenames.push_back("opencv_data\\hand_20.jpg");
	filenames.push_back("opencv_data\\hand_21.jpg");
	filenames.push_back("opencv_data\\hand_22.jpg");
	filenames.push_back("opencv_data\\hand_23.jpg");
	filenames.push_back("opencv_data\\hand_24.jpg");
	filenames.push_back("opencv_data\\hand_25.jpg");
	filenames.push_back("opencv_data\\hand_26.jpg");
	filenames.push_back("opencv_data\\hand_27.jpg");
	filenames.push_back("opencv_data\\hand_28.jpg");
	filenames.push_back("opencv_data\\hand_29.jpg");
	filenames.push_back("opencv_data\\hand_30.jpg");
	filenames.push_back("opencv_data\\hand_31.jpg");
	filenames.push_back("opencv_data\\hand_32.jpg");
	filenames.push_back("opencv_data\\hand_33.jpg");
	filenames.push_back("opencv_data\\hand_34.jpg");
	filenames.push_back("opencv_data\\hand_35.jpg");
	filenames.push_back("opencv_data\\hand_36.jpg");
#elif 0
	filenames.push_back("opencv_data\\simple_hand_01.jpg");
	filenames.push_back("opencv_data\\simple_hand_02.jpg");
	filenames.push_back("opencv_data\\simple_hand_03.jpg");
	filenames.push_back("opencv_data\\simple_hand_04.jpg");
	filenames.push_back("opencv_data\\simple_hand_05.jpg");
	filenames.push_back("opencv_data\\simple_hand_06.jpg");
	filenames.push_back("opencv_data\\simple_hand_07.jpg");
	filenames.push_back("opencv_data\\simple_hand_08.jpg");
	filenames.push_back("opencv_data\\simple_hand_09.jpg");
	filenames.push_back("opencv_data\\simple_hand_10.jpg");
	filenames.push_back("opencv_data\\simple_hand_11.jpg");
	filenames.push_back("opencv_data\\simple_hand_12.jpg");
	filenames.push_back("opencv_data\\simple_hand_13.jpg");
#endif

	const std::string windowName1("skin color filtering - input image");
	const std::string windowName2("skin color filtering - filtered image 1");
	const std::string windowName3("skin color filtering - filtered image 2");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName3, cv::WINDOW_AUTOSIZE);

	// for initialization
	const local::GmmSkinColorModel &gmmSkinColorModel = local::GmmSkinColorModel::getInstance();

	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat &img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat mask1;
		{
			const bool is_uniform_daylight = true;
			const double &startTime = (double)cv::getTickCount();
			local::PeerSkinColorModel::filter(img, is_uniform_daylight, mask1);
			const double &elapsedTime = ((double)cv::getTickCount() - startTime) / cv::getTickFrequency();
			std::cout << elapsedTime << ", ";
		}

		cv::Mat mask2;
		{
			const float theta = 1.0f;
			const double &startTime = (double)cv::getTickCount();
			gmmSkinColorModel.filter(img, theta, mask2);
			const double &elapsedTime = ((double)cv::getTickCount() - startTime) / cv::getTickFrequency();
			std::cout << elapsedTime << std::endl;
		}

		cv::imshow(windowName1, img);
		cv::Mat filtered_img;
		img.copyTo(filtered_img, mask1 > 0);
		cv::imshow(windowName2, filtered_img);
		img.copyTo(filtered_img, mask2 > 0);
		cv::imshow(windowName3, filtered_img);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
	cv::destroyWindow(windowName3);
}

void adaptive_skin_color_filtering()
{
	const int imageWidth = 640, imageHeight = 480;

	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "fail to open vision sensor" << std::endl;
		return;
	}

	//
	const std::string windowName1("skin color filtering - input image");
	const std::string windowName2("skin color filtering - filtered image");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	//
	const int samplingDivider = 1;
	const int morphingMethod = CvAdaptiveSkinDetector::MORPHING_METHOD_ERODE_DILATE;
	CvAdaptiveSkinDetector filter(samplingDivider, morphingMethod);

	cv::Mat gray, frame, frame2, img, filtered_img, filter_mask;
	for (;;)
	{
#if 1
		capture >> frame;
#else
		capture >> frame2;

		if (frame2.cols != imageWidth || frame2.rows != imageHeight)
		{
			//cv::resize(frame2, frame, cv::Size(imageWidth, imageHeight), 0.0, 0.0, cv::INTER_LINEAR);
			cv::pyrDown(frame2, frame);
		}
		else frame = frame2;
#endif

		frame.copyTo(img);
		img.copyTo(filtered_img);

		if (filter_mask.empty())
			filter_mask = cv::Mat::zeros(img.size(), CV_8UC1);

		const std::clock_t &clock = std::clock();
		filter.process(&(IplImage)img, &(IplImage)filter_mask);  // skin color detection
		std::cout << "elapsed time: " << (std::clock() - clock) << std::endl;

		filtered_img.setTo(cv::Scalar(0,255,0), filter_mask);

		cv::imshow(windowName1, img);
		cv::imshow(windowName2, filtered_img);

		const int key = cv::waitKey(1);
		if (27 == key) break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

}  // namespace local
}  // unnamed namespace

void skin_color_filtering()
{
	local::skin_color_filtering();
	//local::adaptive_skin_color_filtering();
}
