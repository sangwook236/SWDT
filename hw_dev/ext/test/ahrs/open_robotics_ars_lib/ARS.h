#if !defined(__SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_H__)
#define __SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_H__ 1


#include "MatrixAlgebra.h"


// Attitude Reference System (ARS).
class ARS
{
protected:
	ARS()
	{
		initialize();
	}
public:
	virtual ~ARS()
	{}

private:
	explicit ARS(const ARS &rhs);  // not implemented.
	ARS & operator=(const ARS &rhs);  // not implemented.

public:
	virtual void initialize();
	// gyroscope : [deg/sec], acceleromter : [g], magnetometer : gauss [G], dt : sampling time.
	virtual dMatrix filter(const double gyro[3], const double accel[3], const double magn[3], const double &dt) = 0;

protected:
	void computeGyroAverage(double average_gyro[3], const double gyro[3], const std::size_t max_count, const double tolerance);

protected:
	// 각속도의 평균 값을 저장한다. 화면에 표시하기 위해 전역변수로 설정함.
	double average_gyro_[3];
	// 가속도센서에서 측정된 중력의 방향으로부터 보정된 roll, pitch, yaw 값. 여기서는 사용하지 않음.
	double adjust_rpy_[3];

private:
	std::size_t average_gyro_count_;
};


#endif  // __SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_H__
