#pragma once

#include "MatrixAlgebra.h"


class Quaternion
{
public:
	Quaternion();
	Quaternion(const double angle_in_rad, const dVector & axis);
	Quaternion(const double s, const double v1, const double v2, const double v3);
	Quaternion(double phi, double theta, double psi);

	Quaternion   operator+(const Quaternion & q)const;
	Quaternion   operator-(const Quaternion & q)const;
	Quaternion   operator*(const Quaternion & q)const;
	Quaternion   operator/(const Quaternion & q)const;
	
	Quaternion   conjugate()const;
	double       norm()const;
	Quaternion & unit(); 
	Quaternion   i()const;
	double       dot(const Quaternion & q)const;

	double       s() const { return s_; }			// Return scalar part.
	void         set_s (const double s){ s_ = s; }	// Set scalar part.
	dVector       v() const { return v_; }			// Return vector part.
	void         set_v (const dVector & v);			// Set vector part.

	dMatrix		 RotationMatrix ();

//private:
public:
	double s_;			// Quaternion scalar part.
	dVector v_;			// Quaternion vector part.
};

Quaternion  operator*(const double c, const Quaternion & rhs);
Quaternion  operator*(const Quaternion & lhs, const double c);
Quaternion  operator/(const double c, const Quaternion & rhs);
Quaternion  operator/(const Quaternion & lhs, const double c);
