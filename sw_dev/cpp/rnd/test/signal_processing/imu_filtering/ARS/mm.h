#pragma once

#include <math.h>

#ifndef M_PI
#define M_PI		3.14159265358979323846	// pi 
#endif

#define _RAD2DEG	(180./M_PI)
#define _DEG2RAD	(M_PI/180.)

inline double DeltaRad (double ang1, double ang2)
{
	double da = ang1 - ang2;
	if (-M_PI < da && da < M_PI) return da;
	else {
		da = fmod (da, 2*M_PI);
		if (M_PI <= da) return da - 2*M_PI;
		else if (da <= -M_PI) return da + 2*M_PI;
		else return da;
	}
	return da;
}

// Use a method described by Box and Muller and discussed in Knuth: 
inline double GaussRand_()
{
	static double v1, v2, s;
    static int phase = 0;
    double z;
	
    if (phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;
			
			v1 = 2 * U1 - 1;
			v2 = 2 * U2 - 1;
			s = v1 * v1 + v2 * v2;
		} while (s >= 1 || s == 0);
		
		z = v1 * sqrt(-2 * log(s) / s);
    }
    else {
		z = v2 * sqrt(-2 * log(s) / s);
	}
	phase = 1 - phase;
	return z;
}

inline double GaussRand()
{
	static double gr[RAND_MAX] = { -1., };

	if (gr[0] == -1.) {
		for (int i=0; i<RAND_MAX; ++i) {
			gr[i] = GaussRand_();
		}
	}
	return gr[rand()];
}

inline double GaussRand2()
{
    static double u, v;
    static int phase = 0;
    double z;
	
    if (phase == 0) {
		u = (rand() + 1.) / (RAND_MAX + 2.);
		v = rand() / (RAND_MAX + 1.);
		z = sqrt(-2 * log(u)) * sin(2 * M_PI * v);
    }
    else {
		z = sqrt(-2 * log(u)) * cos(2 * M_PI * v);
	}
    phase = 1 - phase;
    return z;
}


// These methods all generate numbers with mean 0 and standard deviation 1. 
// (To adjust to another distribution, multiply by the standard deviation 
// and add the mean.) Method 1 is poor ``in the tails'' (especially if NSUM is small), 
// but methods 2 and 3 perform quite well. See the references for more information. 

// Exploit the Central Limit THeorem (``law of large numbers'') and 
// add up several uniformly distributed random numbers: 

inline double GaussRand3()
{
#define NSUM 25

    double z = 0;
    int i;
    
	for (i = 0; i < NSUM; ++i) {
		z += (double)rand() / RAND_MAX;
	}
    z -= NSUM / 2.0;
    z /= sqrt(NSUM / 12.0);
	
    return z;

	// (Don't overlook the sqrt(NSUM / 12.) correction, although it's easy 
	// to do so accidentally, especially when NSUM is 12.) 
}

inline int INTEGER (const double a)
{
	// return (long)floor (a + 0.5);
	return (0 < a)? (int)(a + 0.5) : (int)(a - 0.5);
}
