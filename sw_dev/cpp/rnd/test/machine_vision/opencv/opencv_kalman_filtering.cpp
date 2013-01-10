//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>


namespace {
namespace local {

inline cv::Point calc_trajectory(const cv::Point2f &center, const double radius, const double angle)
{
    return center + cv::Point2f((float)std::cos(angle), (float)-std::sin(angle)) * (float)radius;
}

inline void draw_cross(cv::Mat &img, const cv::Point &center, const cv::Scalar &color, const int d)
{
	cv::line(img, cv::Point(center.x - d, center.y - d), cv::Point(center.x + d, center.y + d), color, 1, CV_AA, 0);
	cv::line(img, cv::Point(center.x + d, center.y - d), cv::Point(center.x - d, center.y + d), color, 1, CV_AA, 0);
}

void kalman_filtering()
{
	const int imageWidth = 640, imageHeight = 480;

	const std::string windowName("Kalman filtering");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	//
	cv::KalmanFilter kalmanFilter(2, 1, 0);

    kalmanFilter.transitionMatrix = *(cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
    cv::setIdentity(kalmanFilter.measurementMatrix);

    cv::setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1e-5));
    cv::setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(1e-1));
	//kalmanFilter.errorCovPre;  // priori error estimate covariance matrix, P'(k): P'(k) = A*P(k-1)*At + Q
    cv::setIdentity(kalmanFilter.errorCovPost, cv::Scalar::all(1));  // posteriori error estimate covariance matrix, P(k): P(k) = (I - K(k)*H) * P'(k)

	//kalmanFilter.statePre;  // predicted state, x'(k): x'(k) = A*x(k-1) + B*u(k)
    cv::randn(kalmanFilter.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));  // corrected state, x(k): x(k) = x'(k) + K(k) * (z(k) - H*x'(k))

	//
	cv::Mat state(2, 1, CV_32F);  // [ phi, delta_phi ]
	cv::Mat processNoise(2, 1, CV_32F);
	cv::Mat measurement = cv::Mat::zeros(1, 1, CV_32F);
	cv::Mat prediction;

	cv::randn(state, cv::Scalar::all(0), cv::Scalar::all(0.1));

	//
	const cv::Point2f center(imageWidth*0.5f, imageHeight*0.5f);
	const float radius = imageWidth / 3.0f;

	cv::Mat img(imageHeight, imageWidth, CV_8UC3);
	for (;;)
	{
		// actual states
		const double &stateAngle = state.at<float>(0);  // phi
		const cv::Point &statePt = calc_trajectory(center, radius, stateAngle);

		// time update (prediction): x(k-1) & P(k-1)  ==>  x-(k) & P-(k)
		{
			prediction = kalmanFilter.predict();
		}

		const double &predictAngle = prediction.at<float>(0);
		const cv::Point &predictPt = calc_trajectory(center, radius, predictAngle);

		// generate measurement
		{
			cv::randn(measurement, cv::Scalar::all(0), cv::Scalar::all(kalmanFilter.measurementNoiseCov.at<float>(0)));
			measurement += kalmanFilter.measurementMatrix * state;
		}

		const double &measureAngle = measurement.at<float>(0);
		const cv::Point &measurePt = calc_trajectory(center, radius, measureAngle);

		// measurement update (correction): x-(k), P-(k) & y_tilde(k)  ==>  K(k), x(k) & P(k)
		{
			kalmanFilter.correct(measurement);

			cv::randn(processNoise, cv::Scalar(0), cv::Scalar::all(std::sqrt(kalmanFilter.processNoiseCov.at<float>(0, 0))));
			state = kalmanFilter.transitionMatrix * state + processNoise;
		}

		//
		img = cv::Scalar::all(0);
		draw_cross(img, statePt, CV_RGB(255,255,255), 3);
		draw_cross(img, measurePt, CV_RGB(255,0,0), 3);
		draw_cross(img, predictPt, CV_RGB(0,255,0), 3);
		cv::line(img, statePt, measurePt, CV_RGB(255,0,0), 3, CV_AA, 0);
		cv::line(img, statePt, predictPt, CV_RGB(255,255,0), 3, CV_AA, 0);

		cv::imshow(windowName, img);

		const unsigned char key = cv::waitKey(100);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void kalman_filtering()
{
	local::kalman_filtering();
}

}  // namespace my_opencv
