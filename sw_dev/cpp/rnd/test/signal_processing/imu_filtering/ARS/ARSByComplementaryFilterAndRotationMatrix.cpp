#include "ARSByComplementaryFilterAndRotationMatrix.h"
#include "mm.h"


// [ref]
//	http://blog.daum.net/pg365/79
//	http://blog.daum.net/pg365/170
//	http://blog.daum.net/pg365/171


void ARSByComplementaryFilterAndRotationMatrix::initialize()
{
	base_type::initialize();

	// 회전행렬을 단위행렬로 초기화 한다.
	R_.unit();
}

dMatrix ARSByComplementaryFilterAndRotationMatrix::filter(const double gyro[3], const double accel[3], const double magn[3], const double &dt)
{
	computeGyroAverage(average_gyro_, gyro, 10000, 10.);

	// 자이로 센서의 바이어스 값을 보정하기 위해 각속도 값에서 각속도 평균값을 뺀다.
	// 그리고 각도의 변화량을 회전행렬로 변환한다.
	dMatrix dR = RotationMatrix(
		_DEG2RAD*(gyro[0] - average_gyro_[0])*dt, 
		_DEG2RAD*(gyro[1] - average_gyro_[1])*dt, 
		_DEG2RAD*(gyro[2] - average_gyro_[2])*dt
	);

	// 회전 행렬의 곱은 두 각도를 더하는 효과를 가진다.
	// R*dR은 R만큼 회전된 좌표계를 dR만큼 더 회전하게 된다.
	R_ = R_ * dR;

	dMatrix g = dMatrix(3, 1, accel);
	dMatrix m = dMatrix(3, 1, magn);

	// accel과 magn은 센서 좌표계를 기준으로 측정된 값이다.
	// 센서의 자세(R)을 곱해서 전역좌표계를 기준으로 한 값으로 바꿔준다.
	g = R_ * g;
	m = R_ * m;

	// 중력가속도 값이 1근처에 있을 때 이득(k1)이 큰 값을 가지도록 한다.
	double l = std::sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]) - 1.;
	double sigma = 0.1;
	double k1 = 0.1 * std::exp(-l * l / (sigma * sigma));

#if defined ADIS_16405
	double k2 = 0.1;

	// 각도의 보정량을 계산한다.
	double dPhi   = g(2,0) ? std::atan(g(1,0) / g(2,0)) : 0.;
	double dTheta = (-1 < g(0,0) && g(0,0) < 1) ? std::asin(-g(0,0) / -1.) : 0.;
	double dPsi   = -std::atan2(m(1,0), m(0,0));
#else
	double k2 = 0.;

	// 각도의 보정량을 계산한다.
	double dPhi   = g(2,0) ? std::atan(g(1,0) / g(2,0)) : 0.;
	double dTheta = (-1 < g(0,0) && g(0,0) < 1) ? std::asin(-g(0,0) / -1.) : 0.;
	double dPsi   = 0.;
#endif

	adjust_rpy_[0] = dPhi;
	adjust_rpy_[1] = dTheta;
	adjust_rpy_[2] = dPsi;

	// 오일러각으로 계산한 보정량을 회전행렬로 변환한다.
	dMatrix aR = RotationMatrix(k1 * dPhi, k1 * dTheta, k2 * dPsi);

	// 계산된 보정량(aR)은 전역좌표계를 기준으로 계산된 값이다.
	// 그러므로 R의 앞에서 aR을 곱해야 한다.
	// R*aR 과 같이 쓰지 않도록 주의한다.
	R_ = aR * R_;

	return R_;
}
