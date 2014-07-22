#if !defined(__SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_BY_EKF_AND_QUATERNION_H__)
#define __SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_BY_EKF_AND_QUATERNION_H__ 1


#include "ARS.h"
#include "Quaternion.h"


class ARSByEKFAndQuaternion : public ARS 
{
public:
	typedef ARS base_type;

public:
	ARSByEKFAndQuaternion()
	: base_type(),
	  x_(), P_(4, 4)
	{}
private:
	virtual ~ARSByEKFAndQuaternion()
	{}

private:
	explicit ARSByEKFAndQuaternion(const ARSByEKFAndQuaternion &rhs);  // not implemented.
	ARSByEKFAndQuaternion & operator=(const ARSByEKFAndQuaternion &rhs);  // not implemented.

public:
	/*virtual*/ void initialize();
	/*virtual*/ dMatrix filter(const double gyro[3], const double accel[3], const double magn[3], const double &dt);

private:
	dMatrix getStateTransitionMatrix(double wx, double wy, double wz, const double &dt) const;
	dMatrix getMeasurementMatrix(const Quaternion &q) const;

private:
	// EKF의 상태 벡터, quaternion.
	Quaternion x_;
	// 상태 벡터의 공분산 행렬.
	dMatrix P_;
};


#endif  // __SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_BY_EKF_AND_QUATERNION_H__
