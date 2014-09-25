#include <iostream>
#include <stdexcept>


namespace {
namespace local {

// [ref] http://robottini.altervista.org/kalman-filter-vs-complementary-filter.
double complementary_filter_1st_for_razor(double newAngle, double newRate, int looptime)
// the first-order complementary filter.
{
	// sparkfun IMU 6DOF Razor.

	// a = tau / (tau + loop time)
	// newAngle = angle measured with atan2 using the accelerometer
	// newRate = angle measured using the gyro
	// looptime = loop time in millis()

	static double x_angleC = 0.0;

	const double tau = 0.075;
	const double dtC = double(looptime) / 1000.0;
	const double a = tau / (tau + dtC);
	x_angleC = a * (x_angleC + newRate * dtC) + (1.0 - a) * (newAngle);
	return x_angleC;
}

// [ref] http://robottini.altervista.org/kalman-filter-vs-complementary-filter.
double complementary_filter_2nd_for_razor(float newAngle, float newRate,int looptime)
// the second-order complementary filter.
{
	// sparkfun IMU 6DOF Razor.

	// newAngle = angle measured with atan2 using the accelerometer
	// newRate = angle measured using the gyro
	// looptime = loop time in millis()

	static double y1 = 0.0;
	static double x_angle2C = 0.0;

	const double k = 10;
	const double dtc2 = double(looptime) / 1000.0;

	const double x1 = (newAngle - x_angle2C) * k * k;
	y1 = dtc2 * x1 + y1;
	const double x2 = y1 + (newAngle - x_angle2C) * 2 * k + newRate;
	x_angle2C = dtc2 * x2 + x_angle2C;

	return x_angle2C;
}

// [ref] http://robottini.altervista.org/kalman-filter-vs-complementary-filter.
double Kalman_filter_for_razor(float newAngle, float newRate,int looptime)
{
	// sparkfun IMU 6DOF Razor.

	// KasBot V1 - Kalman filter module

	static double x_angle = 0.0;
	static double x_bias = 0.0;
	static double P_00 = 0.0, P_01 = 0.0, P_10 = 0.0, P_11 = 0.0;

	const double Q_angle = 0.01; //0.001
	const double Q_gyro = 0.0003; //0.003
	const double R_angle = 0.01; //0.03

	// newAngle = angle measured with atan2 using the accelerometer
	// newRate = angle measured using the gyro
	// looptime = loop time in millis()

	const double dt = double(looptime) / 1000;
	x_angle += dt * (newRate - x_bias);
	P_00 += -dt * (P_10 + P_01) + Q_angle * dt;
	P_01 += -dt * P_11;
	P_10 += -dt * P_11;
	P_11 += Q_gyro * dt;

	const double y = newAngle - x_angle;
	const double S = P_00 + R_angle;
	const double K_0 = P_00 / S;
	const double K_1 = P_10 / S;

	x_angle += K_0 * y;
	x_bias += K_1 * y;
	P_00 -= K_0 * P_00;
	P_01 -= K_0 * P_01;
	P_10 -= K_1 * P_00;
	P_11 -= K_1 * P_01;

	return x_angle;
}

}  // namespace local
}  // unnamed namespace

namespace my_sparkfun {

}  // namespace my_sparkfun

int sparkfun_main(int argc, char *argv[])
{
	throw std::runtime_error("not yet implemented");

	return 0;
}
