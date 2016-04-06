#include "../openhpe_lib/tnc.h"
#include "../openhpe_lib/hand_data.h"
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <iostream>


cv::Scalar refineSegments(const cv::Mat &img, const cv::Mat &mask, cv::Mat &dst, std::vector<cv::Point> &contour, std::vector<cv::Point> &second_contour, cv::Point2i &previous);

namespace {
namespace local {

double rad_to_deg = 180.0 / CV_PI;

cv::Mat laplacian_mtx(int N, bool closed_poly)
{
	cv::Mat A = cv::Mat::zeros(N, N, CV_64FC1);
	cv::Mat d = cv::Mat::zeros(N, 1, CV_64FC1);
	
    //## endpoints
	A.at<double>(0,1) = 1;
	d.at<double>(0,0) = 1;
	
	A.at<double>(N-1,N-2) = 1;
	d.at<double>(N-1,0) = 1;
	
    //## interior points
	for (int i = 1; i <= N - 2; ++i)
	{
        A.at<double>(i, i-1) = 1;
        A.at<double>(i, i+1) = 1;
		
        d.at<double>(i, 0) = 0.5;
	}
	
	cv::Mat Dinv = cv::Mat::diag(d);
	
	return cv::Mat::eye(N, N, CV_64FC1) - Dinv * A;
}

void calc_laplacian(cv::Mat &X, cv::Mat &Xlap)
{
	static cv::Mat lapX = laplacian_mtx(X.rows, false);
	// a feeble attempt to save up in memory allocation.. in 99.9% of the cases this if fires
	if (lapX.rows != X.rows)
		lapX = laplacian_mtx(X.rows, false);
	
	cv::Mat _X;	//handle non-64UC2 matrices
	if (X.type() != CV_64FC2)
	{
		X.convertTo(_X, CV_64FC2);
	}
	else
	{
		_X = X;
	}
	
	std::vector<cv::Mat> v;
	cv::split(_X, v);
	v[0] = v[0].t() * lapX.t();
	v[1] = v[1].t() * lapX.t();
	cv::merge(v, Xlap);
	
	Xlap = Xlap.t();
}

//ad-hoc rotation matrix
inline cv::Mat rotationMat(double a)
{
	double ca = std::cos(a), sa = std::sin(a);
	return (cv::Mat_<double>(2, 2) << ca , sa , -sa , ca);
}

inline cv::Point2d getHandOrigin(HAND_DATA &h)
{
	return h.origin + 40 * (h.origin_offset - cv::Point2d(0.5, 0.5));
}
					  
#define HALF_PI 1.57079633

//get the positiong of the finger's tip, and all joints on the way
cv::Point2d newTip(FINGER_DATA &f, HAND_DATA &h, std::vector<cv::Point2d> &out_joints)
{
	cv::Mat _newTip = (cv::Mat_<double>(1,2) << f.origin_offset.x, f.origin_offset.y);
	_newTip *= rotationMat((h.a - 0.5) * HALF_PI);  // hand angle
	
	cv::Mat vM = (cv::Mat_<double>(1,2) << 1, 0);  // unit vector
	vM *= rotationMat(f.a) * rotationMat((h.a - 0.5) * HALF_PI);  // intial angle
	
	out_joints.push_back(*((cv::Point2d *)(_newTip.data)));  // save first joint
	
	for (int i = 0; i < f.joints_a.size(); ++i)
	{
		vM *= rotationMat(f.joints_a[i]);  // angle of joint
		_newTip += vM * f.joints_d[i] * h.size;  // step forward
		out_joints.push_back(*((cv::Point2d *)(_newTip.data)));  // save joint
	}
	
	_newTip.at<cv::Point2d>(0, 0) += getHandOrigin(h);  // move to offset to get coords in real-world axes
	
	return *((cv::Point2d *)(_newTip.data));
}

void mapVecToData(double X[], HAND_DATA &h)
{
	double *d = X;
	h.a = d[0];
	h.origin_offset.x = d[1];
	h.origin_offset.y = d[2];
	//return;
	
	int counter = 3;
	for (int i = 0; i < 5; ++i)
	{
		//h.fingers[i].a = d[counter++];
		for (int j = 0; j < h.fingers[i].joints_d.size(); ++j)
		{
			//h.fingers[i].joints_a[j] = d[counter++];
			h.fingers[i].joints_d[j] = d[counter++];
		}
	}
	
	//std::cout << "map vec to data: "<< std::endl;
	//for (int i = 0; i < SIZE_OF_HAND_DATA; ++i)
	//{
	//	std::cout << i << ": " << d[i] << std::endl;
	//}
}

void mapDataToVec(double X[], HAND_DATA &h)
{
	double *d = X;
	d[0] = h.a;
	d[1] = h.origin_offset.x;
	d[2] = h.origin_offset.y;
	//return;
	
	int counter = 3;
	for (int i = 0; i < 5; ++i)
	{
		//d[counter++] = h.fingers[i].a;
		for (int j = 0; j < h.fingers[i].joints_d.size(); ++j)
		{
			//d[counter++] = h.fingers[i].joints_a[j];
			d[counter++] = h.fingers[i].joints_d[j];
		}
	}

	//std::cout << "map data to vec: "<< std::endl;
	//for (int i = 0; i < SIZE_OF_HAND_DATA; ++i)
	//{
	//	std::cout << i << ": " << d[i] << std::endl;
	//}
}

cv::Mat hand_template_img;
cv::VideoWriter writer;


static double calc_Energy(DATA_FOR_TNC &d, DATA_FOR_TNC &orig_d)
{
	double _sum = 0.0;
	
	//external energy: closness of tips ot target points
	cv::vector<cv::Point2d> tmp;
	cv::Mat tips(5, 1, CV_64FC2);
	
	cv::Point2d hand_origin = getHandOrigin(d.hand);
	
	for (int j = 0; j < 5; ++j)
	{
		tmp.clear();
		FINGER_DATA f = d.hand.fingers[j];
		cv::Point2d _newTip = newTip(f, d.hand, tmp);

		//double closest = DBL_MAX;
		//for (int i = 0; i < d.targets.size(); ++i)
		//{
		//	double dst = cv::norm(d.targets[i] - _newTip);
		//	if (dst < closest) closest = dst;
		//}
		//_sum += closest;

		// Check each joint (and tip) to see if they are inside the blob or outise
		for (int i = 0; i < tmp.size(); ++i)
		{
			double ds = pointPolygonTest(d.contour, tmp[i] + hand_origin, true);
			ds += 5;
			ds = 1 * ((ds < 0) ? -1 : 1) * (ds * ds);  // quadratic
			_sum -= (ds > 0) ? 0 : (300 * ds);
			
			// add some midway points..
			if (i > 1)
			{
				cv::Point2d midp = tmp[i] + tmp[i-1];
				midp.x /= 2.0; midp.y /= 2.0;
				ds = pointPolygonTest(d.contour, midp + hand_origin, true);
				ds += 5;
				ds = 1 * ((ds < 0) ? -1 : 1) * (ds * ds);  // quadratic
				_sum -= (ds > 0) ? 0 : (300 * ds);				
			}
		}
		
		tips.at<cv::Point2d>(j, 0) = _newTip;
	}
	
	// distances between tips
/*
	cv::Mat D = repeat(tips, 1, 5) - repeat(tips.t(), 5, 1);
	cv::Mat affin(5, 5, CV_64FC1);
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < 5; ++j)
		{
			affin.at<double>(i,j) = cv::norm(D.at<cv::Point2d>(i, j));		
		}
	}
	
	double lda = 1000000000000.0 / cv::determinant(affin);
	//std::cout << "std::log(cv::determinant(affin)) " << lda << std::endl;
	_sum += lda;
*/	
	// internal enevrgy: 
	//	- tips and joints can't go too near each other
	//	- fingers should be lazy: joints angles strive to 0
	
	// lazyness of fingers
	std::vector<double> _angles;
/*
	for (int j = 0; j < 5; ++j)
	{
		FINGER_DATA f = d.hand.fingers[j];
		FINGER_DATA of = orig_d.hand.fingers[j];
		//_angles.push_back(f.a - of.a);
		for (int i = 0; i < f.joints_d.size(); ++i)
		{
			//_angles.push_back(f.joints_a[i] - of.joints_a[i]);
			_angles.push_back(f.joints_d[i] - of.joints_d[i]);
		}
	}
*/
	_angles.push_back(d.hand.a - orig_d.hand.a);
	_sum += 10000 * cv::norm(cv::Mat(_angles));
	
	// count how many black pixels there are inside the palm, to help it stay in the middle
	int nz = 0;
	try
	{
		cv::Mat _tmp(d.hand.palm_size, d.hand.palm_size, CV_8UC1, cv::Scalar(0));
		int h_ps = d.hand.palm_size / 2;
		cv::circle(_tmp, cv::Point(h_ps,h_ps), h_ps, cv::Scalar(255), CV_FILLED);
		
		cv::Mat blobC = hand_template_img(
			cv::Range(MAX((int)floor(hand_origin.y - h_ps), 0), MIN((int)floor(hand_origin.y + h_ps), hand_template_img.rows - 1)),
			cv::Range(MAX((int)floor(hand_origin.x - h_ps), 0), MIN((int)floor(hand_origin.x + h_ps), hand_template_img.cols - 1))
		);
		
		if (blobC.size() == _tmp.size())
		{
			cv::Mat(blobC ^ _tmp).copyTo(_tmp, _tmp);  // xor
			nz = cv::countNonZero(_tmp);
		}
	}
	catch (const cv::Exception)
	{
	}
	_sum += nz * 1000;
	
	if (_sum < 0) return 0;
	return _sum;
}

int showstate(DATA_FOR_TNC &d, int waittime)
{
	cv::Mat img; //(200, 200, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::cvtColor(hand_template_img, img, CV_GRAY2BGR);
	
	cv::Point2d hand_origin = getHandOrigin(d.hand);
 	cv::circle(img, hand_origin, d.hand.palm_size/2, cv::Scalar(255, 150, 0), CV_FILLED);
	
	std::vector<cv::Point2d> joints; 
	for (int j = 0; j < 5; ++j)
	{
		joints.clear();
		
		// calc new joints and tip
		cv::Point2d _newTipt = newTip(d.hand.fingers[j], d.hand, joints);
		cv::circle(img, _newTipt, 5, cv::Scalar(255, 0, 150), 2);
		
		cv::Mat jm(joints);
		jm += cv::Scalar(hand_origin.x, hand_origin.y);
		
		cv::line(img, joints[0], hand_origin, cv::Scalar(0,150,255), 2);
		
		for (int i = 0; i < joints.size(); ++i)
		{
			if (i < joints.size() - 1)
				cv::line(img, joints[i], joints[i+1], cv::Scalar(0, 0, 255), 2);
			
			cv::circle(img, joints[i], 3, cv::Scalar(255, 0, 0), 2);
		}		
	}
	
	std::cout << "origin " << hand_origin.x << "," << hand_origin.y << std::endl;

/*
	for (int i  =0; i < d.targets.size(); ++i)
	{
		cv::circle(img, d.targets[i], 3, cv::Scalar(0, 244, 0), 2);
	}
	
	std::vector<cv::Point> ctr(d.contour.rows);
	cv::Mat ctrm(ctr);
	d.contour.convertTo(ctrm, CV_32SC2);
	std::vector<std::vector<cv::Point> > ctrs;
	ctrs.push_back(ctr);
	cv::drawContours(img, ctrs, -1, Scv::calar(255, 0, 0), 1);
*/	
	cv::imshow("state", img);
	writer << img;
	if (waittime >= 0) return cv::waitKey(waittime); 
	else return -1;
}

static tnc_function my_f;
#define EPSILON 0.1

static int my_f(double x[], double *f, double g[], void *state)
{
	DATA_FOR_TNC *d_ptr = (DATA_FOR_TNC *)state;
	DATA_FOR_TNC new_data = *d_ptr;

	mapVecToData(x, new_data.hand);
	
	*f = calc_Energy(new_data, *d_ptr);
	
	//showstate(new_data, 30);
	
	// calc gradients
	{
		double _x[SIZE_OF_HAND_DATA];
		
		for (int i = 0;i < SIZE_OF_HAND_DATA; ++i)
		{
			memcpy(_x, x, sizeof(double) * SIZE_OF_HAND_DATA);
			_x[i] = _x[i] + EPSILON;
			mapVecToData(_x, new_data.hand);
			double E_epsilon = calc_Energy(new_data, *d_ptr);
			g[i] = ((E_epsilon - *f) / EPSILON);
		}
	}
	
	return 0;
}

void initialize_hand_data(DATA_FOR_TNC &d, const cv::Mat &mymask)
{
/*
	{
		cv::FileStorage fs("./data/object_tracking/fist_handpoints.yaml", cv::FileStorage::READ);
		cv::Mat hand_points;
		cv::Scalar hand_points_midp;
		fs["points"] >> hand_points;
		fs["points_midp_x"] >> hand_points_midp[0];
		fs["points_midp_y"] >> hand_points_midp[1];
		fs["blob"] >> hand_template_img;
		
		{
			d.targets = std::vector<cv::Point2d>(hand_points.rows);
			cv::Mat target_points2dm(d.targets);
			cv::Mat(hand_points + hand_points_midp).convertTo(target_points2dm, CV_64FC2);
		}
		d.hand.origin = cv::Point2d(hand_points_midp[0] + 0, hand_points_midp[1] + 0);
		
		mymask.copyTo(hand_template_img);
		
		std::vector<vector<cv::Point> > contours;
		cv::Mat _tmp;
		//hand_template_img.copyTo(_tmp);
		cv::GaussianBlur(hand_template_img, _tmp, cv::Size(25, 25), 9.0);
		_tmp = (_tmp > 125);
		cv::imshow("mask", _tmp);
		
		cv::findContours(_tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		cv::vector<std::Point> approxCurve;
		approxPolyDP(Mat(contours[0]), approxCurve, 3.0, true);
		Mat(approxCurve).copyTo(d.contour);
		
		cv::Mat lap;
		calc_laplacian(d.contour, lap);
		d.targets.clear();
		for (int i = 0; i < lap.rows; ++i)
		{
			cv::Point2d l = lap.at<cv::Point2d>(i, 0);
			if (cv::norm(l) > 10.0)
			{
				cv::Point p = d.contour.at<cv::Point>(i, 0);
				//std::cout << p.x << "," << p.y << ":" << l.x << "," << l.y << "(" << cv::norm(l) << ")" << std::endl;
				d.targets.push_back(p);
			}
		}
		
		for (int y = 0; y < hand_template_img.rows; ++y)
		{
			uchar *ptr = hand_template_img.ptr<uchar>(y);
			for (int x = 0; x < hand_template_img.cols; ++x)
			{
				double ds = pointPolygonTest(d.contour, cv::Point(x,y), true);
				
				ptr[x] = (ds > 255.0) ? 255 : ds;
			}
		}
	}
*/
	
	std::vector<cv::Point> contour, second_contour;
	mymask.copyTo(hand_template_img);
	cv::Point tmpPoint = (!d.initialized) ? cv::Point2d(-1, -1) : d.hand.origin;
	refineSegments(cv::Mat(), mymask, hand_template_img, contour, second_contour, tmpPoint);
	d.hand.origin = tmpPoint;

	if (contour.size() > 0)
	{
		std::vector<cv::Point> approxCurve;
		approxPolyDP(cv::Mat(contour), approxCurve, 3.0, true);
		cv::Mat(approxCurve).copyTo(d.contour);
	}

/*
	cv::Mat lap;
	calc_laplacian(d.contour, lap);
	d.targets.clear();
	for (int i = 0; i < lap.rows; ++i)
	{
		cv::Point2d l = lap.at<cv::Point2d>(i, 0);
		if (cv::norm(l) > 10.0)
		{
			cv::Point p = d.contour.at<cv::Point>(i, 0);
			d.targets.push_back(p);
		}
	}
*/

	d.hand.origin_offset = cv::Point2d(0.5,0.5);
	d.hand.a = (d.initialized) ? d.hand.a /* .8 + 0.1*/ : 0.5;	// interpolate from last time or reset to .5
	
	// reset the joints lengths to maximum
	for (int i = 0; i < 5; ++i)
	{
		d.hand.fingers[i].joints_d.assign(1, (i < 4) ? 0.89 : 0.40);
	}
	
	if (!d.initialized)
	{
		d.hand.size = 60;
		d.hand.palm_size = 80; //CV_PI/8;
		
		for (int i = 0; i < 4; ++i)
		{
			double a = -(13 * CV_PI / 16) + i * (CV_PI * 6 / 32);  // finger's base angle in respest to center palm
			cv::Mat v(cv::Point2d(d.hand.size * 11 / 16,0));  // length from center palm to finger base
			v = v.t() * rotationMat(a);
			
			//std::cout << a << "," << ((double *)v.data)[0] << "," << ((double *)v.data)[1] << std::endl;
			
			d.hand.fingers[i].origin_offset = *((cv::Point2d *)v.data);
			d.hand.fingers[i].joints_a.assign(1, 0.0);
			//d.hand.fingers[i].joints_d.assign(3, 0.29);
			d.hand.fingers[i].a = -(19 * CV_PI / 32) + i * (CV_PI / 16);  // finger's angle in respect to base
		}
		//toe..
		{
			cv::Mat v(cv::Point2d(d.hand.size * 6 / 8, 0));
			v = v.t() * rotationMat(CV_PI * 3 / 16);
			
			d.hand.fingers[4].origin_offset = *((cv::Point2d *)v.data);
			d.hand.fingers[4].joints_a.assign(1, 0.0);
			//d.hand.fingers[4].joints_d.assign(2, 1.0 / 4.0);
			d.hand.fingers[4].a = -CV_PI / 4;
		}
		
		cv::Scalar _m = cv::mean(cv::Mat(d.targets));
		d.hand.origin = cv::Point2d(_m.val[0], _m.val[1]);
		
		d.initialized = true;
	}
}

static void _onMouse(int event, int x, int y, int flags, void *userdata)
{
	if (event == CV_EVENT_LBUTTONUP)
	{
		DATA_FOR_TNC *_h = (DATA_FOR_TNC *)userdata;
		_h->hand.origin = cv::Point(x, y);
		showstate(*_h, 1);
	}
}

DATA_FOR_TNC d;

void estimateHand(cv::Mat &mymask)
{	
	double _x[SIZE_OF_HAND_DATA] = { 0 };
	cv::Mat X(1, SIZE_OF_HAND_DATA, CV_64FC1, _x);
	double f;
	cv::Mat gradients(cv::Size(SIZE_OF_HAND_DATA, 1), CV_64FC1, cv::Scalar(0));
	
	cv::namedWindow("state");
	
	initialize_hand_data(d, mymask);
	
	mapDataToVec((double *)X.data, d.hand);
	
	simple_tnc(SIZE_OF_HAND_DATA, (double *)X.data, &f, (double*)gradients.data, my_f, (void *)&d, 1, 0);
	
	mapVecToData((double *)X.data, d.hand);

	showstate(d, -1);
	
	d.hand.origin = getHandOrigin(d.hand); //move to new position
}

}  // namespace local
}  // unnamed namespace

namespace my_openhpe {

// [ref] ${OPENHPE_HOME}/main.cpp
void example()
{
	local::initialize_hand_data(local::d, cv::Mat::zeros(cv::Size(640,480), CV_8UC1));
	local::d.hand.origin = cv::Point(320, 240);
	local::showstate(local::d, 0);
	//	return 1;

	cv::VideoCapture capture("./data/object_tracking/output.avi");
	if (capture.isOpened() == false)
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	local::writer.open("./data/object_tracking/estimator.avi", CV_FOURCC('x', 'v', 'i', 'd'), 15.0, cv::Size(640, 480));

	cv::Mat img;

	while (true)
	{
		capture >> img;

		cv::Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);
		local::estimateHand(gray);

		const int c = cv::waitKey(30);
		if (c == 27) break;
		else if (c == 'p') cv::waitKey(0);
	}

	capture.release();
}

}  // namespace my_openhpe
