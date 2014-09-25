#include "ARSByComplementaryFilterAndQuaternion.h"
#include "mm.h"


// [ref]
//	http://blog.daum.net/pg365/81.
//	http://blog.daum.net/pg365/172.


void ARSByComplementaryFilterAndQuaternion::initialize()
{
	base_type::initialize();

	// 단위 쿼터니언으로 초기화 한다.
	Q_.s_ = 1.;
	Q_.v_[0] = 0.;
	Q_.v_[1] = 0.;
	Q_.v_[2] = 0.;
}

dMatrix ARSByComplementaryFilterAndQuaternion::filter(const double gyro[3], const double accel[3], const double magn[3], const double &dt)
{
	computeGyroAverage(average_gyro_, gyro, 10000, 10.);

	// 자이로 센서의 바이어스 값을 보정하기 위해 각속도 값에서 각속도 평균값을 뺀다.
	// 그리고 각도의 변화량을 
	Quaternion dQ(
		_DEG2RAD*(gyro[0] - average_gyro_[0])*dt, 
		_DEG2RAD*(gyro[1] - average_gyro_[1])*dt, 
		_DEG2RAD*(gyro[2] - average_gyro_[2])*dt
	);
	
	// 현재 센서의 자세를 나타내는 쿼터니언(Q_)에 자세 변화량 쿼터니언을 곱해서
	// 자세의 변화를 업데이트 한다.
	//Q_ = Q_ * dQ;

	// 센서가 움직이지 않는 상황에서 항상 중력 가속도는 아래 방향(-z)으로 작용하고 있기
	// 때문에, 센서가 움직이지 않을 때는 가속도 벡터(g)는 중력과 동일하다. 그래서
	// 가속도 벡터의 자세로부터 현재 자세 Q_의 roll과 pitch를 보정할 수 있다.
	dVector g0 = dVector3(0., 0., -1.);
	Quaternion Qa(0, accel[0], accel[1], accel[2]);

	// g는 센서 좌표계를 기준으로 한 벡터이므로, 이들을 전역 좌표계 기준으로 표시하기 
	// 위해서는 Q_을 곱해서 전역 좌표계 기준으로 변환한다.
	Quaternion Qg = Q_ * Qa * Q_.i();

	// 전역 좌표계에서 가속도 센서가 측정한 중력 방향(Qg)과 이상적인 중력 방향(g0)이 이루는 
	// normal vector(ng)와 사잇각(alpha)를 구한다.
	dVector ng = Cross(Qg.v(), g0);
	double alpha = std::asin(Norm2(ng) / Norm2(Qg.v()));

	// 중력으로 찾은 각도의 오차를 업데이트 하는 비율(이득)을 정한다. 
	// 중력 벡터의 크기가 1.근처일 때 이득이 커야하고 1.에서 멀어질 수록 이득이 적어야 한다.
	double l = std::sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]) - 1.;
	double sigma = 0.1;
	double Kg = 0.1 * std::exp(-l * l / (sigma * sigma));

	Q_ = Quaternion(Kg * alpha, ng) * Q_;

#if defined ADIS_16405
	{
		// 지자기는 항상 일정한 방향으로 작용하고 있기 때문에, 지자기 센서의 지자기 
		// 벡터(m)로는 yaw를 보정할 수 있다.
		Quaternion Qm(0, magn[0], magn[1], magn[2]);

		// m은 센서 좌표계를 기준으로 한 벡터이므로, 이들을 전역 좌표계 기준으로 표시하기 
		// 위해서는 Q_을 곱해서 전역 좌표계 기준으로 변환한다.
		Qm = Q_ * Qm * Q_.i();

		dVector mn (0., 0., 1.);
		double alpha = -std::atan2(Qm.v()[1], Qm.v()[0]);
		const double Ka = 0.1;
		
		Q_ = Quaternion(Ka * alpha, mn) * Q_;
	}
#endif

	return Q_.RotationMatrix();
}
