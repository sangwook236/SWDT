#define SIZE_OF_HAND_DATA 8

#include <opencv2/opencv.hpp>

typedef struct finger_data
{
	cv::Point2d origin_offset;  // base or finger relative to center hand
	double a;  // angle
	std::vector<double> joints_a;  // angles of joints
	std::vector<double> joints_d;  // bone length
} FINGER_DATA;

typedef struct hand_data
{
	FINGER_DATA fingers[5];  // fingers
	double a;  // angle of whole hand
	cv::Point2d origin;  //center of palm
	cv::Point2d origin_offset;  //offset from center for optimization
	double size;  //relative size of hand = length of a finger
	double palm_size;
} HAND_DATA;

typedef struct data_for_tnc
{
	std::vector<cv::Point2d> targets;  // points to reach
	HAND_DATA hand;
	cv::Mat contour;
	cv::Mat hand_blob;  // 8bit (1bit) mask of the hand
	
	bool initialized;
} DATA_FOR_TNC;