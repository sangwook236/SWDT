/*
 * Copyright (C) 2009-2010 Andre Schulz, Florian Jung, Sebastian Hartte,
 *						   Daniel Trick, Christan Wojek, Konrad Schindler,
 *						   Jens Ackermann, Michael Goesele
 * Copyright (C) 2008-2009 Christopher Evans <chris.evans@irisys.co.uk>, MSc University of Bristol
 *
 * This file is part of SURFGPU.
 *
 * SURFGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SURFGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SURFGPU.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CUDA_NONMAXCUDA_CU
#define CUDA_NONMAXCUDA_CU

#ifdef __DEVICE_EMULATION__
#	include <stdio.h>
#endif
#include "../fasthessian_cudaipoint.h"
#include "../defines.h"

/////////////////////////////////////////////////////////
// Definitions and constants
/////////////////////////////////////////////////////////

#define MVAL9(M,y,x) M[y*3+x]
#define MVAL12(M,y,x) M[y*4+x]
#define MVAL27(M,y,x,z) M[z*9+y*3+x]

__constant__ int dc_lobe_map[16];
__constant__ float eps = 0.0000002499f;

__device__ void interpolateExtremum(float *det, fasthessian_cudaIpoint *points,
									int octv, int intvl, int r, int c,
									int init_sample, int intervals,
									int i_width, int i_height, int step);
__device__ void interpolateStep(float *m_det, int octv, int intvl, int r, int c,
								float *xi, float *xr, float *xc);
__device__ int d_interpolatePos(float *H, int octv, int intvl, int r, int c,
								float* xi, float* xr, float* xc);
__device__ int interpolateFindmax(float *dH, float *dHH,
								  float* xi, float* xr, float* xc);
__device__ void interpolateFirstDerivH(float *H, float *dH);
__device__ void interpolateSecondDerivH(float *H, float *dHH);
__device__ float getVal_d(float *m_det, int o, int i, int c, int r,
						  int intervals, int i_width, int i_height);

__device__ int atomicCounter;

//-------------------------------------------------------

/**	\brief Find interest points using the given determinants and perform NMS
 *	\param g_det device pointer to determinants
 *	\param g_points device pointer to save the interest points to
 *	\param i_width g_det width
 *	\param i_height g_det height
 *	\param i_widthStep number of elements in a row of g_det
 *	\param intervals number of intervals to compute
 *	\param o octave to compute
 *	\param step number of pixels skipped in X/Y
 *	\param steps_i
 *
 *	Computation is done pixel-wise. One thread processes one pixel.
 *	i_widthStep is currently unused.
 *	Recommended execution configuration:
 *	  Thread block: { 16, 8 }
 *	  Block grid  : { number of pixels to process in X, same in Y }
 */
__global__ void
nonMaxCUDA(float *g_det, unsigned int i_width, unsigned int i_height,
		   size_t i_widthStep, fasthessian_cudaIpoint *g_points,
		   int intervals, int o, int step, int steps_i, int border,
		   float thres, int init_sample)
{
	//Get current interval
	const int interval_size = gridDim.x / steps_i;
	const int i = ((blockIdx.x / interval_size) * 3) + 1;

	//Get current column and row
	const int c = ((((blockIdx.x % interval_size) * blockDim.x) + threadIdx.x) * 3 * step) + border;
	const int r = ((blockIdx.y * blockDim.y + threadIdx.y) * 3 * step) + border;

	if (c >= i_width - border || r >= i_height - border || i >= intervals - 1)
		return;

	int i_max = -1, r_max = -1, c_max = -1;
	float max_val = 0.0f;

	// Scan the pixels in this block to find the local extremum.
	for (int ii = i; ii < min(i + 3, intervals - 1); ii += 1) {
		for (int rr = r; rr < min(r + 3 * step, i_height - border); rr += step) {
			for (int cc = c; cc < min(c + 3 * step, i_width - border); cc += step) {
				float val = getVal_d(g_det, o, ii, cc, rr, intervals, i_width, i_height);

				// record the max value and its location
				if (val > max_val) {
					max_val = val;
					i_max = ii;
					r_max = rr;
					c_max = cc;
				}
			}
		}
	}

	int extremum = 0;

	// Bounds check
	if (!(i_max - 1 < 0 || i_max + 1 > intervals - 1
		  || c_max - step < 0 || c_max + step > i_width
		  || r_max - step < 0 || r_max + step > i_height))
	{
		// Check for maximum
		for (int ii = i_max - 1; ii <= i_max + 1; ++ii)
			for (int cc = c_max - step; cc <= c_max + step; cc += step)
				for (int rr = r_max - step; rr <= r_max + step; rr += step) {
					if (ii != 0 || cc != 0 || rr != 0) {
						if (getVal_d(g_det, o, ii, cc, rr, intervals, i_width, i_height) > max_val) {
							return;
						} else {
							extremum = 1;
						}
					}
				}
	}

	// Check the block extremum is an extremum across boundaries.
	if (extremum && max_val > thres && i_max != -1) {
		interpolateExtremum(g_det, g_points, o, i_max, r_max, c_max,
							init_sample, intervals, i_width, i_height, step);
	}

}

/////////////////////////////////////////////////////////
// Device functions
/////////////////////////////////////////////////////////

//! Non Maximal Suppression function
__device__ void
interpolateExtremum(float *det, fasthessian_cudaIpoint *points,
					int octv, int intvl, int r, int c,
					int init_sample, int intervals, int i_width, int i_height,
					int step)
{
	float xi = 0.0f, xr = 0.0f, xc = 0.0f;
	float H[27];

	int q = 0;

	for (int j = -step; j <= step; j += step)
	{
		for (int k = -step; k <= step; k += step)
		{
			for (int i = -1; i <= 1; i++)
			{
				H[q] = getVal_d(det, octv, intvl+i, c+j, r+k, intervals, i_width, i_height);
				q++;
			}
		}
	}

#ifdef INTERPOLATION_ENABLED
	if(!d_interpolatePos(H, octv, intvl, r, c, &xi, &xr, &xc))
	{
		return;
	}
#endif

	//cuprintf("Interpolate i:%d, r:%d, c:%d => ", intvl, r, c);
	//cuprintf("i:%f, r:%f, c:%f\n", xi, xr, xc);

	if (fabsf(xi) < 0.5f
		&& fabsf(xr) < 0.5f
		&& fabsf(xc) < 0.5f)
	{
		int index = atomicAdd(&atomicCounter, 1);
		if (index < ((i_width * i_height) / IMG_SIZE_DIVISOR))
		{
			unsigned int lobe = dc_lobe_map[octv * intervals + intvl];
			unsigned int idx = lobe * i_width * i_height + (r * i_width + c);
			float res = det[idx];
			points[index].x = c + step * xc;
			points[index].y = r + step * xr;
			points[index].scale = 1.2f/9.0f * (3.0f * ((1 << (octv+1)) * (intvl + xi + 1.0f) + 1.0f));
			points[index].laplacian = (res >= 0.0f ? 1.0f : -1.0f);
		}
	}
}

//-------------------------------------------------------

__device__ int
d_interpolatePos(float *H, int octv, int intvl, int r, int c,
				 float* xi, float* xr, float* xc)
{
	float dh[3];
	float ddh[9];

	interpolateFirstDerivH(H, dh);
	interpolateSecondDerivH(H, ddh);

	return interpolateFindmax(dh, ddh, xi, xr, xc);
}

//-------------------------------------------------------

__device__ void
divR(float *R, int u, float divVal)
{
	for (int i = 0; i < 4; i++)
	{
		MVAL12(R, u, i) = MVAL12(R, u, i) / divVal;
	}
}

//-------------------------------------------------------

__device__ int
interpolateFindmax(float *dH, float *dHH, float* xi, float* xr, float* xc)
{
	// Gaussian elimination (need not fully invert the matrix)

	float R[12];
	int elimorder[3];
	int redeq[2];

	MVAL12(R,0,0) = MVAL9(dHH,0,0);
	MVAL12(R,0,1) = MVAL9(dHH,0,1);
	MVAL12(R,0,2) = MVAL9(dHH,0,2);
	MVAL12(R,0,3) = -dH[0];
	MVAL12(R,1,0) = MVAL9(dHH,1,0);
	MVAL12(R,1,1) = MVAL9(dHH,1,1);
	MVAL12(R,1,2) = MVAL9(dHH,1,2);
	MVAL12(R,1,3) = -dH[1];
	MVAL12(R,2,0) = MVAL9(dHH,2,0);
	MVAL12(R,2,1) = MVAL9(dHH,2,1);
	MVAL12(R,2,2) = MVAL9(dHH,2,2);
	MVAL12(R,2,3) = -dH[2];

	// scale equations so that coefficient for x is 1
	for (int u = 0; u < 3; u++) {
		if (fabsf(MVAL12(R, u, 0)) > eps)
			divR(R, u, MVAL12(R, u, 0));
	}

	// eliminate x from 2 equations
	if (fabsf(MVAL12(R,0,0)) > eps) {
		if (fabsf(MVAL12(R,1,0)) > eps) {
			for (int i = 0; i < 4; i++)
				MVAL12(R, 1, i) = MVAL12(R, 1, i) - MVAL12(R, 0, i);
		}

		if (fabsf(MVAL12(R, 2, 1)) > eps) {
			for (int i = 0; i < 4; i++)
				MVAL12(R, 2, i) = MVAL12(R, 2, i) - MVAL12(R, 0, i);
		}

		elimorder[0] = 9;
		elimorder[1] = 9;
		elimorder[2] = 0;
		redeq[0] = 1;
		redeq[1] = 2;

	} else if (fabsf(MVAL12(R,1,0)) > eps) {
		if (fabsf(MVAL12(R,2,0)) > eps) {
			for (int i = 0; i < 4; i++)
				MVAL12(R, 2, i) = MVAL12(R, 2, i) - MVAL12(R, 1, i);
		}
		elimorder[0] = 9;
		elimorder[1] = 9;
		elimorder[2] = 1;
		redeq[0] = 0;
		redeq[1] = 2;
	} else if (fabsf(MVAL12(R,2,0)) > eps) {
		elimorder[0] = 9;
		elimorder[1] = 9;
		elimorder[2] = 2;
		redeq[0] = 0;
		redeq[1] = 1;
	} else {
		return 0;
	}

	// eliminate y from 1 equation (if still necessary)
	if (fabsf(MVAL12(R,redeq[0], 1)) > eps
		&& fabsf(MVAL12(R,redeq[1], 1)) > eps) //leading coeffs not already 0
	{
		for (int i = 0; i < 4; i++) {
			divR(R, redeq[0], MVAL12(R,redeq[0],1));
		}

		for (int i = 0; i < 4; i++) {
			divR(R, redeq[1], MVAL12(R,redeq[1],1));
		}

		for (int i = 0; i < 4; i++) {
			MVAL12(R,redeq[1],i) = MVAL12(R,redeq[1],i) - MVAL12(R,redeq[0],i);
		}

		elimorder[0] = redeq[1];
		elimorder[1] = redeq[0];

	} else if (fabsf(MVAL12(R,redeq[0], 1)) > eps) {
		elimorder[0] = redeq[1];
		elimorder[1] = redeq[0];
	} else if (fabsf(MVAL12(R,redeq[1], 1)) > eps) {
		elimorder[0] = redeq[0];
		elimorder[1] = redeq[1];
	} else {
		return 0;
	}

	float pos[3];

	pos[elimorder[0]] = MVAL12(R,elimorder[0],3) / MVAL12(R,elimorder[0],2);
	pos[elimorder[1]] = (MVAL12(R,elimorder[1],3) - MVAL12(R,elimorder[1],2) * pos[elimorder[0]]) / MVAL12(R,elimorder[1],1);
	pos[elimorder[2]] = (MVAL12(R,elimorder[2],3) - MVAL12(R,elimorder[2],2) * pos[elimorder[0]] - MVAL12(R,elimorder[2],1) * pos[elimorder[1]]) / MVAL12(R,elimorder[2],0);

	*xc = pos[2];
	*xi = pos[1];
	*xr = pos[0];

	return 1;
}

//-------------------------------------------------------

__device__ void
interpolateSecondDerivH(float *H, float *dHH)
{
	MVAL9(dHH,0,0) = -2.0f * MVAL27(H,1,1,1) + MVAL27(H,0,1,1) + MVAL27(H,2,1,1);
	MVAL9(dHH,1,1) = -2.0f * MVAL27(H,1,1,1) + MVAL27(H,1,2,0) + MVAL27(H,1,2,1);
	MVAL9(dHH,2,2) = -2.0f * MVAL27(H,1,1,1) + MVAL27(H,1,1,0) + MVAL27(H,1,1,2);

	MVAL9(dHH,0,1) = .25f * (MVAL27(H,0,0,1)+MVAL27(H,2,2,1) - MVAL27(H,0,2,1) - MVAL27(H,2,0,1));
	MVAL9(dHH,0,2) = .25f * (MVAL27(H,0,1,0)+MVAL27(H,2,1,2) - MVAL27(H,0,1,2) - MVAL27(H,2,1,0));
	MVAL9(dHH,1,2) = .25f * (MVAL27(H,1,0,0)+MVAL27(H,1,2,2) - MVAL27(H,1,0,2) - MVAL27(H,1,2,0));
	MVAL9(dHH,1,0) = MVAL9(dHH,0,1);
	MVAL9(dHH,2,0) = MVAL9(dHH,0,2);
	MVAL9(dHH,2,1) = MVAL9(dHH,1,2);
}

//-------------------------------------------------------

__device__ void
interpolateFirstDerivH(float *H, float *dH)
{
	dH[0] = 0.5f * (MVAL27(H, 2, 1, 1)-MVAL27(H, 0, 1, 1));
	dH[1] = 0.5f * (MVAL27(H, 1, 2, 1)-MVAL27(H, 1, 0, 1));
	dH[2] = 0.5f * (MVAL27(H, 1, 1, 2)-MVAL27(H, 1, 1, 0));
}

//-------------------------------------------------------

//! Return the value of the approximated determinant of hessian
__device__ float
getVal_d(float *m_det, int o, int i, int c, int r,
		 int intervals, int i_width, int i_height)
{
	unsigned int lobe = dc_lobe_map[o * intervals + i];
	unsigned int idx = lobe * (i_width * i_height) + (r * i_width + c);
	return fabsf(m_det[idx]);
}

#endif /* CUDA_NONMAXCUDA_CU */
