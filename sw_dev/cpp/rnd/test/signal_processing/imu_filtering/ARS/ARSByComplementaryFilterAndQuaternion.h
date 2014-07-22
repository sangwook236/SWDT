#if !defined(__SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_BY_COMPLEMENTARY_FILTER_AND_QUATERNION_H__)
#define __SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_BY_COMPLEMENTARY_FILTER_AND_QUATERNION_H__ 1


#include "ARS.h"
#include "Quaternion.h"


class ARSByComplementaryFilterAndQuaternion : public ARS 
{
public:
	typedef ARS base_type;

public:
	ARSByComplementaryFilterAndQuaternion()
	: base_type(),
	  Q_(0, 0, 0)
	{}
private:
	virtual ~ARSByComplementaryFilterAndQuaternion()
	{}

private:
	explicit ARSByComplementaryFilterAndQuaternion(const ARSByComplementaryFilterAndQuaternion &rhs);  // not implemented.
	ARSByComplementaryFilterAndQuaternion & operator=(const ARSByComplementaryFilterAndQuaternion &rhs);  // not implemented.

public:
	/*virtual*/ void initialize();
	/*virtual*/ dMatrix filter(const double gyro[3], const double accel[3], const double magn[3], const double &dt);

private:
	// roll, pitch, yaw 값을 가지는 quaternion.
	Quaternion Q_;
};


#endif  // __SIGNAL_PROCESSING__IMU_FILTERING__ARS__ARS_BY_COMPLEMENTARY_FILTER_AND_QUATERNION_H__
