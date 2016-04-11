#include "adisusbz/AdisUsbz.h"
#include <iostream>
#include <vector>
#if defined(WIN32) || defined(_WIN32)
#define _USE_MATH_DEFINES
#endif
#include <math.h>


namespace {
namespace local {

const char ADIS16350_SUPPLY_OUT = 0x02;
const char ADIS16350_XGYRO_OUT = 0x04;
const char ADIS16350_YGYRO_OUT = 0x06;
const char ADIS16350_ZGYRO_OUT = 0x08;
const char ADIS16350_XACCL_OUT = 0x0A;
const char ADIS16350_YACCL_OUT = 0x0C;
const char ADIS16350_ZACCL_OUT = 0x0E;
const char ADIS16350_XTEMP_OUT = 0x10;
const char ADIS16350_YTEMP_OUT = 0x12;
const char ADIS16350_ZTEMP_OUT = 0x14;
const char ADIS16350_AUX_ADC = 0x16;

const double ADIS16350_SUPLY_SCALE_FACTOR = 1.8315e-1;  // binary, [V]
const double ADIS16350_GYRO_SCALE_FACTOR = 0.07326;  // 2's complement, [deg/sec]
														//const double ADIS16350_GYRO_SCALE_FACTOR =	0.07326 * boost::math::constants::pi<double>() / 180.0;  // 2's complement, [rad/sec]
const double ADIS16350_ACCL_SCALE_FACTOR = 2.522e-3;  // 2's complement, [g]
const double ADIS16350_TEMP_SCALE_FACTOR = 0.1453;  // 2's complement, [deg]
const double ADIS16350_ADC_SCALE_FACTOR = 0.6105e-3;  // binary, [V]

const double deg2rad = M_PI / 180.0;
const double lambda = 36.368 * deg2rad;  // latitude [rad]
const double phi = 127.364 * deg2rad;  // longitude [rad]
const double h = 71.0;  // altitude: 71 ~ 82 [m]
const double sin_lambda = std::sin(lambda);
const double sin_2lambda = std::sin(2.0 * lambda);

// [ref] wikipedia: Gravity of Earth
// (latitude, longitude, altitude) = (lambda, phi, h) = (36.368, 127.364, 71.0)
// g(lambda, h) = 9.780327 * (1 + 0.0053024 * sin(lambda)^2 - 0.0000058 * sin(2 * lambda)^2) - 3.086 * 10^-6 * h
const double REF_GRAVITY = 9.780327 * (1.0 + 0.0053024 * sin_lambda*sin_lambda - 0.0000058 * sin_2lambda*sin_2lambda) - 3.086e-6 * local::h;  // [m/sec^2]
const double REF_ANGULAR_VEL = 7.292115e-5;  // [rad/sec]

bool read_adis_data(AdisUsbz &adis, std::vector<double> &measuredAccel, std::vector<double> &measuredGyro)
{
	const short rawXGyro = adis.ReadInt14(ADIS16350_XGYRO_OUT) & 0x3FFF;
	const short rawYGyro = adis.ReadInt14(ADIS16350_YGYRO_OUT) & 0x3FFF;
	const short rawZGyro = adis.ReadInt14(ADIS16350_ZGYRO_OUT) & 0x3FFF;
	const short rawXAccel = adis.ReadInt14(ADIS16350_XACCL_OUT) & 0x3FFF;
	const short rawYAccel = adis.ReadInt14(ADIS16350_YACCL_OUT) & 0x3FFF;
	const short rawZAccel = adis.ReadInt14(ADIS16350_ZACCL_OUT) & 0x3FFF;

	// [m/sec^2]
	measuredAccel[0] = ((rawXAccel & 0x2000) == 0x2000 ? (rawXAccel - 0x4000) : rawXAccel) * ADIS16350_ACCL_SCALE_FACTOR * REF_GRAVITY;
	measuredAccel[1] = ((rawYAccel & 0x2000) == 0x2000 ? (rawYAccel - 0x4000) : rawYAccel) * ADIS16350_ACCL_SCALE_FACTOR * REF_GRAVITY;
	measuredAccel[2] = ((rawZAccel & 0x2000) == 0x2000 ? (rawZAccel - 0x4000) : rawZAccel) * ADIS16350_ACCL_SCALE_FACTOR * REF_GRAVITY;

	// [rad/sec]
	measuredGyro[0] = ((rawXGyro & 0x2000) == 0x2000 ? (rawXGyro - 0x4000) : rawXGyro) * ADIS16350_GYRO_SCALE_FACTOR * deg2rad;
	measuredGyro[1] = ((rawYGyro & 0x2000) == 0x2000 ? (rawYGyro - 0x4000) : rawYGyro) * ADIS16350_GYRO_SCALE_FACTOR * deg2rad;
	measuredGyro[2] = ((rawZGyro & 0x2000) == 0x2000 ? (rawZGyro - 0x4000) : rawZGyro) * ADIS16350_GYRO_SCALE_FACTOR * deg2rad;

	return true;
}

bool test_adis_usbz(AdisUsbz &adis, const std::size_t loopCount)
{
	std::vector<double> measuredAccel(3, 0.0), measuredGyro(3, 0.0);

	std::size_t loop = 0;
	while (loop++ < loopCount)
	{
		if (!read_adis_data(adis, measuredAccel, measuredGyro))
			return false;

		std::cout <<
			measuredAccel[0] << ", " << measuredAccel[1] << ", " << measuredAccel[2] << " ; " <<
			measuredGyro[0] << ", " << measuredGyro[1] << ", " << measuredGyro[2] << std::endl;
	}

	return true;
}

void calculate_calibrated_acceleration(const std::vector<double> &accel_calibration_param, const std::vector<double> &lg, std::vector<double> &a_calibrated)
{
	const double &b_gx = accel_calibration_param[0];
	const double &b_gy = accel_calibration_param[1];
	const double &b_gz = accel_calibration_param[2];
	const double &s_gx = accel_calibration_param[3];
	const double &s_gy = accel_calibration_param[4];
	const double &s_gz = accel_calibration_param[5];
	const double &theta_gyz = accel_calibration_param[6];
	const double &theta_gzx = accel_calibration_param[7];
	const double &theta_gzy = accel_calibration_param[8];

	const double &l_gx = lg[0];
	const double &l_gy = lg[1];
	const double &l_gz = lg[2];

	const double tan_gyz = std::tan(theta_gyz);
	const double tan_gzx = std::tan(theta_gzx);
	const double tan_gzy = std::tan(theta_gzy);
	const double cos_gyz = std::cos(theta_gyz);
	const double cos_gzx = std::cos(theta_gzx);
	const double cos_gzy = std::cos(theta_gzy);

	const double g_x = (l_gx - b_gx) / (1.0 + s_gx);
	const double g_y = tan_gyz * (l_gx - b_gx) / (1.0 + s_gx) + (l_gy - b_gy) / ((1.0 + s_gy) * cos_gyz);
	const double g_z = (tan_gzx * tan_gyz - tan_gzy / cos_gzx) * (l_gx - b_gx) / (1.0 + s_gx) +
		((l_gy - b_gy) * tan_gzx) / ((1.0 + s_gy) * cos_gyz) + (l_gz - b_gz) / ((1.0 + s_gz) * cos_gzx * cos_gzy);

	a_calibrated[0] = g_x;
	a_calibrated[1] = g_y;
	a_calibrated[2] = g_z;
}

void calculate_calibrated_gyro(const std::vector<double> &gyro_calibration_param, const std::vector<double> &lw, std::vector<double> &w_calibrated)
{
	const double &b_wx = gyro_calibration_param[0];
	const double &b_wy = gyro_calibration_param[1];
	const double &b_wz = gyro_calibration_param[2];

	const double &l_wx = lw[0];
	const double &l_wy = lw[1];
	const double &l_wz = lw[2];

	const double w_x = l_wx - b_wx;
	const double w_y = l_wy - b_wy;
	const double w_z = l_wz - b_wz;

	w_calibrated[0] = w_x;
	w_calibrated[1] = w_y;
	w_calibrated[2] = w_z;
}

void initialize_gravity(AdisUsbz &adis, const std::size_t numInitializationStep, const std::vector<double> &accelCalibrationParam, const std::vector<double> &gyroCalibrationParam, std::vector<double> &initialGravity, std::vector<double> &initialGyro)
{
	double accel_x_sum = 0.0, accel_y_sum = 0.0, accel_z_sum = 0.0;
	//double gyro_x_sum = 0.0, gyro_y_sum = 0.0, gyro_z_sum = 0.0;

	std::vector<double> measuredAccel(3, 0.0), measuredGyro(3, 0.0);
	std::vector<double> calibratedAccel(3, 0.0), calibratedGyro(3, 0.0);
	for (std::size_t i = 0; i < numInitializationStep; ++i)
	{
		read_adis_data(adis, measuredAccel, measuredGyro);

		calculate_calibrated_acceleration(accelCalibrationParam, measuredAccel, calibratedAccel);
		//calculate_calibrated_gyro(gyroCalibrationParam, measuredGyro, calibratedGyro);

		accel_x_sum += calibratedAccel[0];
		accel_y_sum += calibratedAccel[1];
		accel_z_sum += calibratedAccel[2];
		//gyro_x_sum += calibratedGyro[0];
		//gyro_y_sum += calibratedGyro[1];
		//gyro_z_sum += calibratedGyro[2];
	}

	initialGravity[0] = accel_x_sum / numInitializationStep;
	initialGravity[1] = accel_y_sum / numInitializationStep;
	initialGravity[2] = accel_z_sum / numInitializationStep;
	//initialGyro[0] = gyro_x_sum / numInitializationStep;
	//initialGyro[1] = gyro_y_sum / numInitializationStep;
	//initialGyro[2] = gyro_z_sum / numInitializationStep;
}

}  // namespace local
}  // unnamed namespace

namespace my_adis {

}  // namespace my_adis

int adis_main(int argc, char *argv[])
{
	// sampling interval
	const double samplingTime = 0.01;  // [sec]

	const std::size_t numInitializationStep = 10000;

	// Initialize ADIS.
	AdisUsbz adis;

#if defined(UNICODE) || defined(_UNICODE)
	if (!adis.Initialize(L"\\\\.\\Ezusb-0"))
#else
	if (!adis.Initialize("\\\\.\\Ezusb-0"))
#endif
	{
		std::cout << "fail to initialize ADISUSBZ" << std::endl;
		return 1;
	}

#if 0
	// Test ADIS.
	const std::size_t loopCount = 100;
	local::test_adis_usbz(adis, loopCount);
#else
	// Initialize gravity.
	std::cout << "set an initial gravity ..." << std::endl;
/*
	// FIXME [fix] >> calibrate parameters of accelerometer and gyro, and calculate initial gravity and gyro.
	std::vector<double> initialGravity(3, 0.0), initialGyro(3, 0.0);
	{
		// Load calibration parameters.
		std::vector<double> accelCalibrationParam(3, 0.0), gyroCalibrationParam(3, 0.0);

		local::initialize_gravity(adis, numInitializationStep, accelCalibrationParam, gyroCalibrationParam, initialGravity, initialGyro);
	}
*/

	// Read data.
	{
		const std::size_t numStep = 10000;

		std::vector<double> measuredAccel(3, 0.0);
		std::vector<double> measuredGyro(3, 0.0);

		std::size_t step = 0;
		while (step < numStep)
		{
			local::read_adis_data(adis, measuredAccel, measuredGyro);

			// apply measurement to calibration and initial gravity and 
		}
	}
#endif

	return 0;
}
