#include "Quaternion.h"
#include <math.h>
#include <assert.h>

#define ROTATE_ZYX

#define EPSILON		0.0000001

Quaternion::Quaternion()
: s_(1.), v_(0., 3)
{
}

Quaternion::Quaternion(const double angle, const dVector & axis)
{
	if(axis.size() != 3) {
		assert (0 && "Quaternion::Quaternion, size of axis != 3");
		return;
	}

	// make sure axis is a unit vector
	v_ = sin(angle/2) * axis/Norm2(axis);
	s_ = cos(angle/2);
}

Quaternion::Quaternion(const double s_in, const double v1, const double v2, const double v3)
: s_(s_in), v_(0., 3)
{
	v_[0] = v1;
	v_[1] = v2;
	v_[2] = v3;
}

Quaternion::Quaternion(double phi, double theta, double psi)
: s_(0.), v_(0., 3)
{
	Quaternion Qx (cos(phi/2),   sin(phi/2), 0,            0         );
	Quaternion Qy (cos(theta/2), 0,          sin(theta/2), 0         );
	Quaternion Qz (cos(psi/2),   0,          0,            sin(psi/2));

#ifdef ROTATE_ZYX	
	*this = Qz*Qy*Qx;
#else
	*this = Qx*Qy*Qz;
#endif
}

inline double sgn (double a) 
{
	return (0. < a) ? 1. : ((a < 0) ? -1. : 0.);
}

Quaternion Quaternion::operator+(const Quaternion & rhs)const
{
	Quaternion q;
	q.s_ = s_ + rhs.s_;
	q.v_ = v_ + rhs.v_;

	return q;
}

Quaternion Quaternion::operator-(const Quaternion & rhs)const
{
	Quaternion q;
	q.s_ = s_ - rhs.s_;
	q.v_ = v_ - rhs.v_;

	return q;
}

Quaternion Quaternion::operator*(const Quaternion & rhs)const
{
	Quaternion q;
	q.s_ = s_ * rhs.s_ - Dot(v_, rhs.v_);
	q.v_ = s_ * rhs.v_ + rhs.s_ * v_ + Cross(v_, rhs.v_);

	return q;
}


Quaternion Quaternion::operator/(const Quaternion & rhs)const
{
	return *this*rhs.i();
}


Quaternion Quaternion::conjugate()const
{
	Quaternion q;
	q.s_ = s_;
	q.v_ = -1.*v_;

	return q;
}

double Quaternion::norm()const 
{ 
	return( sqrt(s_*s_ + Dot(v_,v_)) );
}

Quaternion & Quaternion::unit()
{
	double tmp = norm();
	if(tmp > EPSILON) {
		s_ = s_/tmp;
		v_ = v_/tmp;
	}
	return *this;
}

Quaternion Quaternion::i()const 
{ 
	return conjugate()/norm();
}

double Quaternion::dot(const Quaternion & q)const
{
	return (s_*q.s_ + v_[0]*q.v_[0] + v_[1]*q.v_[1] + v_[2]*q.v_[2]);
}

void Quaternion::set_v(const dVector & v)
{
	if(v.size() == 3) {
		v_ = v;
	}
	else {
		assert (0 && "Quaternion::set_v: input has a wrong size.");
	}
}

dMatrix	Quaternion::RotationMatrix ()
{
	dMatrix R(3, 3);

	R(0, 0) = 2.*(s_*s_ + v_[0]*v_[0]) - 1.;	R(0, 1) = 2.*(v_[0]*v_[1] - s_*v_[2]);		R(0, 2) = 2.*(v_[0]*v_[2] + s_*v_[1]);
	R(1, 0) = 2.*(v_[0]*v_[1] + s_*v_[2]);		R(1, 1) = 2.*(s_*s_ + v_[1]*v_[1]) - 1.;	R(1, 2) = 2.*(v_[1]*v_[2] - s_*v_[0]);
	R(2, 0) = 2.*(v_[0]*v_[2] - s_*v_[1]);		R(2, 1) = 2.*(v_[1]*v_[2] + s_*v_[0]);		R(2, 2) = 2.*(s_*s_ + v_[2]*v_[2]) - 1.;

	return R;
}

Quaternion operator*(const double c, const Quaternion & q)
{
	Quaternion out;
	out.set_s(q.s() * c);
	out.set_v(q.v() * c);
	return out;
}

Quaternion operator*(const Quaternion & q, const double c)
{
	return operator*(c, q);
}

Quaternion operator/(const double c, const Quaternion & q)
{
	Quaternion out;
	out.set_s(q.s() / c);
	out.set_v(q.v() / c);
	return out;
}

Quaternion operator/(const Quaternion & q, const double c)
{
	return operator/(c, q);
}
