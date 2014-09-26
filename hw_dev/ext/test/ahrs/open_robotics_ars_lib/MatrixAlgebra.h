#pragma	once
#include "matrix/cmatrix"
#include <assert.h>

using namespace techsoft;

typedef std::valarray<double>		dVector;
typedef techsoft::matrix<double>	dMatrix;

typedef techsoft::mslice			mSlice;
typedef std::slice					vSlice;


inline dMatrix RotationMatrix (double phi, double theta, double psi)
{
	double sin_phi = sin(phi),		cos_phi = cos(phi);
	double sin_tht = sin(theta),	cos_tht = cos(theta);
	double sin_psi = sin(psi),		cos_psi = cos(psi);

	dMatrix A(3,3);
/*
	// 변환 순서: Rx(phi) x Ry(theta) x Rz(psi)
	A(0,0) =  cos_tht*cos_psi;								A(0,1) = -cos_tht*sin_psi;								A(0,2) =  sin_tht;
	A(1,0) =  sin_phi*sin_tht*cos_psi + cos_phi*sin_psi;	A(1,1) = -sin_phi*sin_tht*sin_psi + cos_phi*cos_psi;	A(1,2) = -sin_phi*cos_tht;
	A(2,0) = -cos_phi*sin_tht*cos_psi + sin_phi*sin_psi;	A(2,1) =  cos_phi*sin_tht*sin_psi + sin_phi*cos_psi;	A(2,2) =  cos_phi*cos_tht;
*/
	// 변환 순서: Rz(psi) x Ry(theta) x Rx(phi)
	A(0,0) =  cos_psi*cos_tht;	A(0,1) = cos_psi*sin_phi*sin_tht - cos_phi*sin_psi;		A(0,2) =  sin_phi*sin_psi + cos_phi*cos_psi*sin_tht;
	A(1,0) =  cos_tht*sin_psi;	A(1,1) = cos_phi*cos_psi + sin_phi*sin_psi*sin_tht;		A(1,2) = cos_phi*sin_psi*sin_tht - cos_psi*sin_phi;
	A(2,0) = -sin_tht;			A(2,1) = cos_tht*sin_phi;								A(2,2) =  cos_phi*cos_tht;

	return A;
}


inline dVector GetOrientation (dMatrix &R)
{
	dVector v(3);

	//  - theta in the range (-pi/2, pi/2)
	v[0] = atan2(R(2,1), R(2,2));
	v[1] = atan2(-R(2,0), sqrt(R(2,1)*R(2,1) + R(2,2)*R(2,2)));
	v[2] = atan2(R(1,0), R(0,0));

	return v;
}

inline dVector GetOrientation2 (dMatrix &R)
{
	dVector v(3);

	//  - theta in the range (pi/2, 3pi/2)
	v[0] = atan2(-R(2,1), -R(2,2));
	v[1] = atan2(-R(2,0), -sqrt(R(2,1)*R(2,1) + R(2,2)*R(2,2)) );
	v[2] = atan2(-R(1,0), -R(0,0));

	return v;
}

inline dVector dVector3 (double v0, double v1, double v2)
{
	dVector V(3);

	V[0] = v0;
	V[1] = v1;
	V[2] = v2;

	return V;
}

inline double Dot (const dVector &v1, const dVector &v2)
{
	if(v1.size() != v2.size())
		assert (0 && "ERROR: Dot(): Inconsistent vector size in Inner Product !");

	double	v = 0.;
	for(unsigned int i=0; i<v1.size(); ++i)
		v += v1[i] * v2[i];

	return v;
}

// dVector의 외적(cross product)을 계산한다.
inline dVector Cross (const dVector &v1, const dVector &v2)
{
	if(v1.size() != 3 || v2.size() != 3)
		assert (0 && "ERROR: Cross(): dVector dimension should be 3 in Cross Product !");

	dVector v(3);
	v[0] = v1[1]*v2[2] - v1[2]*v2[1];
	v[1] = v1[2]*v2[0] - v1[0]*v2[2];
	v[2] = v1[0]*v2[1] - v1[1]*v2[0];

	return v;
}

// Vector의 크기를 2-norm으로 계산한다
inline double Norm2 (const dVector &v)
{
	double s = 0.;
	for (int i=0; i<(int)v.size(); ++i) {
		s += v[i]*v[i];
	}
	return sqrt(s);
}

inline dMatrix MakeMatrix (const dVector &v)
{
	dMatrix M(v.size (), 1);

	for(unsigned int i=0; i<v.size(); ++i) {
		M[(int)i][0] = v[i];
	}
	return M;
}
