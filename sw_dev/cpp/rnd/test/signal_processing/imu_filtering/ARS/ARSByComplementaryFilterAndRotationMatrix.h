#if !defined(__SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_BY_COMPLEMENTARY_FILTER_AND_ROTATION_MATRIX_H__)
#define __SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_BY_COMPLEMENTARY_FILTER_AND_ROTATION_MATRIX_H__ 1


#include "ARS.h"


class ARSByComplementaryFilterAndRotationMatrix : public ARS 
{
public:
	typedef ARS base_type;

public:
	ARSByComplementaryFilterAndRotationMatrix()
	: base_type(),
	  R_(3, 3)
	{}
private:
	virtual ~ARSByComplementaryFilterAndRotationMatrix()
	{}

private:
	explicit ARSByComplementaryFilterAndRotationMatrix(const ARSByComplementaryFilterAndRotationMatrix &rhs);  // not implemented.
	ARSByComplementaryFilterAndRotationMatrix & operator=(const ARSByComplementaryFilterAndRotationMatrix &rhs);  // not implemented.

public:
	/*virtual*/ void initialize();
	/*virtual*/ dMatrix filter(const double gyro[3], const double accel[3], const double magn[3], const double &dt);

private:
	// roll, pitch, yaw 값을 가지는 3x3 회전행렬.
	dMatrix R_;
};


#endif  // __SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_BY_COMPLEMENTARY_FILTER_AND_ROTATION_MATRIX_H__
