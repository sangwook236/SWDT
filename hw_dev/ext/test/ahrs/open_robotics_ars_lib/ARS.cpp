//#include "stdafx.h"
#include "ARS.h"


void ARS::initialize()
{
	for (int i = 0; i < 3; ++i)
	{
		average_gyro_[i] = 0.0;
		adjust_rpy_[i] = 0.0;
	}

	average_gyro_count_ = 0;
}

void ARS::computeGyroAverage(double average_gyro[3], const double gyro[3], const std::size_t max_count, const double tolerance)
{
	// 센서가 정지해 있더라도 각속도계(gyroscope)에서 읽은 값은 0의 값을 가지지 않는다.
	// 대부분 특정한 값으로 바이어스 되어있는데, 이 값을 0으로 보정하기 위하여
	// 센서가 움직이지 않을 때 각속도 값의 평균을 구한다.

	double d0 = gyro[0] - average_gyro[0];
	double d1 = gyro[1] - average_gyro[1];
	double d2 = gyro[2] - average_gyro[2];

	if (!average_gyro_count_ || ( 
		-tolerance < d0 && d0 < tolerance && 
		-tolerance < d1 && d1 < tolerance && 
		-tolerance < d2 && d2 < tolerance)
		)
	{
		average_gyro[0]  = (average_gyro[0]*average_gyro_count_ + gyro[0])/(average_gyro_count_+1);
		average_gyro[1]  = (average_gyro[1]*average_gyro_count_ + gyro[1])/(average_gyro_count_+1);
		average_gyro[2]  = (average_gyro[2]*average_gyro_count_ + gyro[2])/(average_gyro_count_+1);
	}

	if (average_gyro_count_ < max_count) ++average_gyro_count_;
}
