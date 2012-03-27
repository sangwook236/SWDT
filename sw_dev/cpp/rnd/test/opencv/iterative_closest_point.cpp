//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/flann/flann.hpp>
#include <iostream>
#include <list>
#include <limits>
#include <ctime>


namespace {

void ShowPoints(cv::Mat &im, cv::Mat &X, const cv::Scalar &c = cv::Scalar(255))
{
	for (int i = 0; i < X.rows; ++i)
	{
		cv::Point p;
		if (X.type() == CV_32SC1 || X.type() == CV_32SC2)
			p = X.at<cv::Point>(i, 0);
		else if (X.type() == CV_32FC1 || X.type() == CV_32FC2)
		{
			const cv::Point2f &_p = X.at<cv::Point2f>(i, 0);
			p.x = (int)_p.x;
			p.y = (int)_p.y;
		}

		cv::circle(im, p, 2, c, CV_FILLED);
	}
}

void ShowLines(cv::Mat &im, cv::Mat &X, cv::Mat &X_bar, const cv::Scalar &c1 = cv::Scalar(0, 255), const cv::Scalar &c2 = cv::Scalar(0, 0, 255))
{
	//cv::Mat im = cv::Mat::zeros(300, 500, CV_8UC3);
	for (int i = 0; i < X.rows; ++i)
	{
		cv::Point p;
		if (X.type() == CV_32SC1 || X.type() == CV_32SC2)
			p = X.at<cv::Point>(i, 0);
		else if (X.type() == CV_32FC1 || X.type() == CV_32FC2)
		{
			const cv::Point2f &_p = X.at<cv::Point2f>(i, 0);
			p.x = (int)_p.x;
			p.y = (int)_p.y;
		}
		cv::circle(im, p, 1, c1, CV_FILLED);

		const cv::Point &p1 = X_bar.at<cv::Point>(i, 0);
		cv::circle(im, p1, 1, c2, CV_FILLED);

		cv::line(im, p, p1, cv::Scalar(0, 0, 255), 1);

		if (i >= 1)
		{
			cv::Point p_tag;
			if (X.type() == CV_32SC1 || X.type() == CV_32SC2)
				p_tag = X.at<cv::Point>(i-1, 0);
			else if (X.type() == CV_32FC1 || X.type() == CV_32FC2)
			{
				const cv::Point2f &_p = X.at<cv::Point2f>(i-1, 0);
				p_tag.x = (int)_p.x;
				p_tag.y = (int)_p.y;
			}
			cv::line(im, p, p_tag, cv::Scalar(255, 150, 0), 1);

			const cv::Point &p1_tag = X_bar.at<cv::Point>(i-1, 0);
			cv::line(im, p1, p1_tag, cv::Scalar(155, 255, 0), 1);
		}
	}
}

void ShowQuery(cv::Mat &destinations, cv::Mat &query, cv::Mat &closest)
{
	cv::Mat im = cv::Mat::zeros(480, 640, CV_8UC3);
	ShowPoints(im, destinations);
	ShowLines(im, query, closest);
	cv::imshow("tmp", im);
	cv::waitKey();
}

float flann_knn(cv::Mat &m_destinations, cv::Mat &m_object, std::vector<int> &ptpairs, std::vector<float> &dists)
{
	// find nearest neighbors using FLANN
	cv::Mat m_indices(m_object.rows, 1, CV_32S);
	cv::Mat m_dists(m_object.rows, 1, CV_32F);

	cv::Mat dest_32f, obj_32f;
	m_destinations.convertTo(dest_32f, CV_32FC2);
	m_object.convertTo(obj_32f, CV_32FC2);

	assert(dest_32f.type() == CV_32F);

	cv::flann::Index flann_index(dest_32f, cv::flann::KDTreeIndexParams(2));  // using 2 randomized kdtrees
	flann_index.knnSearch(obj_32f, m_indices, m_dists, 1, cv::flann::SearchParams(64));

	int *indices_ptr = m_indices.ptr<int>(0);
	//float *dists_ptr = m_dists.ptr<float>(0);
	for (int i = 0; i < m_indices.rows; ++i)
	{
		ptpairs.push_back(indices_ptr[i]);
	}

	dists.resize(m_dists.rows);
#if defined(__GNUC__)
    cv::Mat dists_mat(dists);
	m_dists.copyTo(dists_mat);
#else
	m_dists.copyTo(cv::Mat(dists));
#endif

	return (float)cv::sum(m_dists)[0];
}

void findBestReansformSVD(cv::Mat &_m, cv::Mat &_d)
{
	cv::Mat m, d;
	_m.convertTo(m, CV_32F);
	_d.convertTo(d, CV_32F);

	cv::Scalar d_bar = cv::mean(d);
	cv::Scalar m_bar = cv::mean(m);
	cv::Mat mc = m - m_bar;
	cv::Mat dc = d - d_bar;

	mc = mc.reshape(1);
	dc = dc.reshape(1);

	cv::Mat H(2, 2, CV_32FC1);
	for (int i = 0; i < mc.rows; ++i)
	{
		cv::Mat mci = mc(cv::Range(i, i+1), cv::Range(0, 2));
		cv::Mat dci = dc(cv::Range(i, i+1), cv::Range(0, 2));
		H = H + mci.t() * dci;
	}

	cv::SVD svd(H);

	cv::Mat R = svd.vt.t() * svd.u.t();
	double det_R = cv::determinant(R);
	if (std::abs(det_R + 1.0) < 0.0001)
	{
		double _tmp[4] = { 1, 0, 0, cv::determinant(svd.vt * svd.u) };
		R = svd.u * cv::Mat(2, 2, CV_32FC1, _tmp) * svd.vt;
	}

#ifdef BTM_DEBUG
	//for some strange reason the debug version of OpenCV is flipping the matrix
	R = -R;
#endif

	float *_R = R.ptr<float>(0);
	cv::Scalar T(d_bar[0] - (m_bar[0]*_R[0] + m_bar[1]*_R[1]), d_bar[1] - (m_bar[0]*_R[2] + m_bar[1]*_R[3]));

	m = m.reshape(1);
	m = m * R;
	m = m.reshape(2);
	m = m + T;  // + m_bar;
	m.convertTo(_m, CV_32S);
}

/*
void findBestTransform(cv::Mat &X, cv::Mat &X_bar)
{
	cv::namedWindow("tmp");
	{
		cv::Mat im = cv::Mat::zeros(300, 500, CV_8UC3);
		ShowLines(im, X, X_bar);
		cv::imshow("tmp", im);
		cv::waitKey(30);
	}

	// shift points to mean point
	const cv::Scalar &xm = cv::mean(X);

	X = X - xm;
	X_bar = X_bar - xm;

	cv::Mat X32f, X_bar_32f;
	X.convertTo(X32f, CV_32F);
	X_bar.convertTo(X_bar_32f, CV_32F);

	cv::Mat A(2 * X.rows, 4, CV_32FC1);
	cv::Mat b(2 * X.rows, 1, CV_32FC1);

	for (int i = 0; i < X.rows; ++i)
	{
		float *Ap = A.ptr<float>(2 * i);
		const cv::Point2f &xi = X32f.at<cv::Point2f>(i, 0);
		const float _A[8] = { xi.x, xi.y, 1, 0, xi.y, -xi.x, 0, 1 };
		memcpy(Ap, _A, sizeof(float) * 8);

		float *bp = b.ptr<float>(2 * i);
		const cv::Point2f &xi_b = X_bar_32f.at<cv::Point2f>(i, 0);
		const float _b[2] = { xi_b.x, xi_b.y };
		memcpy(bp, _b, sizeof(float) * 2);
	}

	// solve linear eq. system: Ax = b  ==>  x = inv(AtA) * (Atb)
	cv::Mat sol = (A.t() * A).inv() * (A.t() * b);

	const float *sd = (float *)sol.data;

	std::cout << "solution: ";
	for (int i = 0; i < 4; ++i) std::cout << sd[i] << ",";
	std::cout << std::endl;

	// 2D totation matrix
	float _R[4] = { sd[0], sd[1], -sd[1], sd[0] };
	cv::Mat R(2, 2, CV_32FC1, _R);

	// transform points
	X32f = X32f.reshape(1);
	cv::Mat Xnew = (X32f) * R.inv();
	Xnew = Xnew.reshape(2);
	Xnew += cv::Scalar(sd[2], sd[3]);

	// restore to original location
	Xnew = Xnew + xm;
	X_bar = X_bar + xm;

	{
		cv::Mat im = cv::Mat::zeros(300, 500, CV_8UC3);
		ShowLines(im, Xnew, X_bar);
		cv::imshow("tmp", im);
		cv::waitKey(30);
	}

	Xnew = Xnew.reshape(2);
	Xnew.convertTo(X, CV_32SC1);
}
*/

void ICP(cv::Mat &X, cv::Mat &destination)
{
	std::vector<int> pair;
	double lastDist = std::numeric_limits<double>::max();
	cv::Mat lastGood;

	bool re_reshape = false;
	if (X.channels() == 2)
	{
		X = X.reshape(1);
		re_reshape = true;
	}
	if (destination.channels() == 2)
	{
		destination = destination.reshape(1);
		re_reshape = true;
	}
	std::vector<float> dists;

	while (true)
	{
		pair.clear();
		dists.clear();

		const double &dist = flann_knn(destination, X, pair, dists);

		if (lastDist <= dist)
		{
			X = lastGood;
			break;  //converged?
		}

		lastDist = dist;
		X.copyTo(lastGood);

		std::cout << "distance: " << dist << std::endl;

		cv::Mat X_bar(X.size(), X.type());
		for (int i = 0; i < X.rows; ++i)
			X_bar.at<cv::Point>(i, 0) = destination.at<cv::Point>(pair[i], 0);

		ShowQuery(destination, X, X_bar);

		X = X.reshape(2);
		X_bar = X_bar.reshape(2);
		findBestReansformSVD(X, X_bar);
		X = X.reshape(1);  // back to 1-channel
	}

	//lastGood.copyTo(X);

	if (re_reshape)
	{
		X = X.reshape(2);
		destination = destination.reshape(2);
	}

	std::cout << "converged" << std::endl;
}

}

void iterative_closest_point()
{
#if 0
	cv::RNG rng(std::time(NULL));

	cv::Mat X(10, 2, CV_32SC1);  // a matrix of point data
	for (int i = 0; i < 10; ++i)
		X.at<cv::Point>(i, 0) = cv::Point(100 + std::sin(((double)i / 10.0) * CV_PI) * 50.0, 100 + std::cos(((double)i / 10.0) * CV_PI) * 50.0);

	cv::Mat destinations(50, 2, X.type());  // a matrix of point data
	rng.fill(destinations, cv::RNG::NORMAL, cv::Scalar(150.0), cv::Scalar(125.0, 50.0));

	ICP(X, destinations);
#else
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
	filenames.push_back("opencv_data\\hand_01.jpg");
	//filenames.push_back("opencv_data\\hand_02.jpg");
	//filenames.push_back("opencv_data\\hand_03.jpg");
	//filenames.push_back("opencv_data\\hand_04.jpg");
	//filenames.push_back("opencv_data\\hand_05.jpg");
	//filenames.push_back("opencv_data\\hand_06.jpg");
	//filenames.push_back("opencv_data\\hand_07.jpg");
	//filenames.push_back("opencv_data\\hand_08.jpg");
	//filenames.push_back("opencv_data\\hand_09.jpg");
	//filenames.push_back("opencv_data\\hand_10.jpg");
	//filenames.push_back("opencv_data\\hand_11.jpg");
	//filenames.push_back("opencv_data\\hand_12.jpg");
	//filenames.push_back("opencv_data\\hand_13.jpg");
	//filenames.push_back("opencv_data\\hand_14.jpg");
	//filenames.push_back("opencv_data\\hand_15.jpg");
	//filenames.push_back("opencv_data\\hand_16.jpg");
	//filenames.push_back("opencv_data\\hand_17.jpg");
	//filenames.push_back("opencv_data\\hand_18.jpg");
	//filenames.push_back("opencv_data\\hand_19.jpg");
	//filenames.push_back("opencv_data\\hand_20.jpg");
	//filenames.push_back("opencv_data\\hand_21.jpg");
	//filenames.push_back("opencv_data\\hand_22.jpg");
	//filenames.push_back("opencv_data\\hand_23.jpg");
	//filenames.push_back("opencv_data\\hand_24.jpg");
	//filenames.push_back("opencv_data\\hand_25.jpg");
	//filenames.push_back("opencv_data\\hand_26.jpg");
	//filenames.push_back("opencv_data\\hand_27.jpg");
	//filenames.push_back("opencv_data\\hand_28.jpg");
	//filenames.push_back("opencv_data\\hand_29.jpg");
	//filenames.push_back("opencv_data\\hand_30.jpg");
	//filenames.push_back("opencv_data\\hand_31.jpg");
	//filenames.push_back("opencv_data\\hand_32.jpg");
#elif 0
	filenames.push_back("opencv_data\\hand_33.jpg");
	filenames.push_back("opencv_data\\hand_34.jpg");
	filenames.push_back("opencv_data\\hand_35.jpg");
	filenames.push_back("opencv_data\\hand_36.jpg");
#endif

	const std::string windowName1("iterative closest point - original");
	const std::string windowName2("iterative closest point - processed");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			//cv::cvtColor(img, gray, CV_RGB2GRAY);

		//
#if 1
		cv::Mat point_cloud_img;
		cv::blur(gray, point_cloud_img, cv::Size(3, 3));

		// run the edge detector on grayscale
		const int lowerEdgeThreshold = 10, upperEdgeThreshold = 30;
		cv::Canny(point_cloud_img, point_cloud_img, lowerEdgeThreshold, upperEdgeThreshold, 3, false);
#else
		//const int ksize = 5;
		const int ksize = CV_SCHARR;  // use Scharr operator
		cv::Mat xgradient, ygradient;
		cv::Sobel(gray, xgradient, CV_32FC1, 1, 0, ksize, 1.0, 0.0);
		cv::Sobel(gray, ygradient, CV_32FC1, 0, 1, ksize, 1.0, 0.0);

		cv::Mat gradient, point_cloud_img;
		cv::magnitude(xgradient, ygradient, gradient);

		const double thresholdRatio = 0.1;
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(gradient, &minVal, &maxVal);
		cv::compare(gradient, minVal + (maxVal - minVal) * thresholdRatio, point_cloud_img, cv::CMP_GT);
#endif
		//
		const int radius = 19;
		const cv::Point center(30, 30);
#if 1
		cv::Mat shape_img(cv::Mat::zeros(90, 60, CV_8UC1));
		cv::line(shape_img, cv::Point(center.x - radius, center.y), cv::Point(center.x - radius, center.y + 50), CV_RGB(255, 255, 255), 1, 8, 0);
		cv::line(shape_img, cv::Point(center.x + radius, center.y), cv::Point(center.x + radius, center.y + 50), CV_RGB(255, 255, 255), 1, 8, 0);
		cv::ellipse(shape_img, center, cv::Size(radius, radius), 0.0, 180.0, 360.0, CV_RGB(255, 255, 255), 1, 8, 0);

		std::vector<cv::Point> shape_points;
		shape_points.reserve(shape_img.rows * shape_img.cols);
		for (int r = 0; r < shape_img.rows; ++r)
			for (int c = 0; c < shape_img.cols; ++c)
				if (shape_img.at<unsigned char>(r, c))
					shape_points.push_back(cv::Point(c, r));
		cv::Mat shape(shape_points, false);  // a matrix of point data
#else
		std::vector<cv::Point> shape_points;
		const cv::Point offset(437, 428);
		for (int i = 50; i >= 0; i -= 2)
			shape_points.push_back(cv::Point(center.x + radius, center.y + i) + offset);
		for (int i = 180; i <= 360; i += 5)
		{
			const double angle = i * CV_PI / 180.0;
			shape_points.push_back(cv::Point((int)cvRound(center.x + radius * std::cos(angle)), (int)cvRound(center.y + radius * std::sin(angle))) + offset);
		}
		for (int i = 0; i <= 50; i += 2)
			shape_points.push_back(cv::Point(center.x - radius, center.y + i) + offset);
		cv::Mat shape(shape_points, false);  // a matrix of point data
#endif

		//
		std::vector<cv::Point> points;
		points.reserve(point_cloud_img.rows * point_cloud_img.cols);
		for (int r = 0; r < point_cloud_img.rows; ++r)
			for (int c = 0; c < point_cloud_img.cols; ++c)
				if (point_cloud_img.at<unsigned char>(r, c))
					points.push_back(cv::Point(c, r));
		cv::Mat point_cloud(points, false);  // a matrix of point data

		ICP(shape, point_cloud);

		//
		cv::imshow(windowName1, shape_img);
		cv::imshow(windowName2, point_cloud_img);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
#endif
}
