#include "ARSByEKFAndQuaternion.h"
#include "mm.h"


// [ref]
//	http://blog.daum.net/pg365/83.
//	http://blog.daum.net/pg365/172.


void ARSByEKFAndQuaternion::initialize()
{
	base_type::initialize();

	// EKF의 상태 벡터와 공분산 행렬을 초기화 한다.
	x_.set_s(1);
	x_.set_v(dVector3(0, 0, 0));

	P_.unit();  // 공분산 행렬은 단위행렬로 초기화
}

dMatrix ARSByEKFAndQuaternion::getStateTransitionMatrix(double wx, double wy, double wz, const double &dt) const
{
	dMatrix A(4, 4);

	wx *= 0.5*dt;
	wy *= 0.5*dt;
	wz *= 0.5*dt;

	A(0,0) = 1;		A(0,1) = wz;	A(0,2) = -wy;	A(0,3) = wx;
	A(1,0) = -wz;	A(1,1) = 1;		A(1,2) = wx;	A(1,3) = wy;
	A(2,0) = wy;	A(2,1) = -wx;	A(2,2) = 1;		A(2,3) = wz;
	A(3,0) = -wx;	A(3,1) = -wy;	A(3,2) = -wz;	A(3,3) = 1;

	return A;
}

dMatrix ARSByEKFAndQuaternion::getMeasurementMatrix(const Quaternion &q) const
{
	const double eps = 1e-10;

	const double e1 = q.v_[0];
	const double e2 = q.v_[1];
	const double e3 = q.v_[2];
	double a = e3*(e1*e1 + e2*e2 + e3*e3);
	double b = std::sqrt(e1*e1 + e2*e2);

	if (-eps < a && a < eps) a = eps;
	if (b < eps) b = eps;

	const double _h[1][4] = { 
		{ (e1*e3*e3 + eps)/(a*b + eps), (e2*e3*e3 + eps)/(a*b + eps), (e3*b + eps)/(a + eps), 0	},
	};

	// observation matrix
	dMatrix H(1, 4, &_h[0][0]);

	return H;
}

dMatrix ARSByEKFAndQuaternion::filter(const double gyro[3], const double accel[3], const double magn[3], const double &dt)
{
	computeGyroAverage(average_gyro_, gyro, 10000, 10.);

	/////////////////////////////////////////////////////////////////////////
	// Predict 과정

	// 자이로 센서의 바이어스 값을 보정하기 위해 각속도 값에서 각속도 평균값을 뺀다.
	dVector w = dVector3(
		_DEG2RAD*(gyro[0] - average_gyro_[0]), 
		_DEG2RAD*(gyro[1] - average_gyro_[1]), 
		_DEG2RAD*(gyro[2] - average_gyro_[2])
	);

	// state transition matrix
	dMatrix A = getStateTransitionMatrix (w[0], w[1], w[2], dt);
	
	double _b[4][3] = {
		{ x_.s_,    -x_.v_[2],  x_.v_[1] },
		{ x_.v_[2],  x_.s_,    -x_.v_[0] },
		{-x_.v_[1],  x_.v_[0],  x_.s_    },
		{-x_.v_[0], -x_.v_[1], -x_.v_[2] },
	};

	dMatrix B(4, 3, &_b[0][0]);
	B = B * 0.5 * dt;

	const double _q[3][3] = { 
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 0, 0, 1 },
	};

	// Covariance matrix of porcess noises
	const dMatrix Q(3, 3, &_q[0][0]);

	// Predicted state estimate
	x_ = x_ * Quaternion(w[0]*dt, w[1]*dt, w[2]*dt);
	// Predicted estimate covariance
	P_ = A * P_ * ~A + B * Q * ~B;

	/////////////////////////////////////////////////////////////////////////
	// Update 과정

	// 센서가 움직이지 않는 상황에서 항상 중력 가속도는 아래 방향(-z)으로 작용하고 있기
	// 때문에, 센서가 움직이지 않을 때는 가속도 벡터(g)는 중력과 동일하다. 그래서
	// 가속도 벡터의 자세로부터 현재 자세 _Q를 보정할 수 있다.
	dVector g0 = dVector3(0., 0., -1.);
	Quaternion a(0, accel[0], accel[1], accel[2]);

	// g는 센서 좌표계를 기준으로 한 벡터이므로, 이들을 전역 좌표계 기준으로 표시하기 
	// 위해서는 _Q을 곱해서 전역 좌표계 기준으로 변환한다.
	Quaternion ag = x_ * a * x_.i();

	// 전역 좌표계에서 가속도 센서가 측정한 중력 방향(Qg)과 이상적인 중력 방향(g0)이 이루는 
	// normal vector(ng)와 사잇각(alpha)를 구한다.
	dVector ng = Cross(ag.v(), g0);
	double alpha = std::asin(Norm2(ng) / Norm2(ag.v()));

	dMatrix H = getMeasurementMatrix(x_);

	// 중력으로 찾은 각도의 오차를 업데이트 하는 비율(이득)을 정한다. 
	// 중력 벡터의 크기가 1.근처일 때 이득이 커야하고 1.에서 멀어질 수록 이득이 적어야 한다.
	double l = std::sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]) - 1.;
	double s = std::exp(-l * l);

	const double _r[1][1] = { 
		{ 1/s },
	};

	const dMatrix R(1, 1, &_r[0][0]);

	// Kalman gain
	dMatrix K = P_ * ~H * !(H * P_ * ~H + R);

	double Ka = K.normF ();

	// Updated state estimate
	x_ = Quaternion(Ka * alpha, ng) * x_;
	x_.unit();

	// Updated estimate covariance
	P_ = P_ - K * H * P_;

	return x_.RotationMatrix ();
}
