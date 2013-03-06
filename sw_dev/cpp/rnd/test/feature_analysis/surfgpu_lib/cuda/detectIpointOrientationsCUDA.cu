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

#ifndef CUDA_DETECTIPOINTORIENTATIONSCUDA_CU
#define CUDA_DETECTIPOINTORIENTATIONSCUDA_CU

#ifdef __DEVICE_EMULATION__
#	include <stdio.h>
#endif
#include "../surf_cudaipoint.h"
#include "common_kernel.h"
#include <math.h>

//--S [] 2013/03/06: Sang-Wook Lee
#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif
//--E [] 2013/03/06: Sang-Wook Lee


 // Texture reference to the integral image; neede by haarXY()
texture<float, 2, cudaReadModeElementType> integralImage;
#include "haarXY.cu"

__constant__ int dc_coord_x[109];
__constant__ int dc_coord_y[109];
__constant__ float dc_gauss_lin[109];

/*
 * This inline function computes the angle for a X,Y value pair
 * and is a verbatim copy of the CPU version.
 */
__device__ float
getAngle(float X, float Y)
{
	float pi = M_PI;

	if (X >= 0.0f && Y >= 0.0f)
		return atanf(Y/X);

	if (X < 0.0f && Y >= 0.0f)
		return pi - atanf(-Y/X);

	if (X < 0.0f && Y < 0.0f)
		return pi + atanf(Y/X);

	if (X >= 0.0f && Y < 0.0f)
		return 2.0f*pi - atanf(-Y/X);

	return 0.0f;
}

/**	\brief Detect orientations of interest points
 *	\param ipoints device pointer to interest points
 *
 *	This kernel runs the entire orientation detection.
 *	Execution configuration:
 *	  Thread block: { 42, 1, 1 }
 *	  Block grid  : { num_ipoints, 1 }
 */
__global__ void
detectIpointOrientiationsCUDA(surf_cudaIpoint* g_ipoints)
{
	surf_cudaIpoint *g_ipt = g_ipoints + blockIdx.x; // Get a pointer to the interest point processed by this block

	// 1. Take all samples required to compute the orientation
	int s = fRound(g_ipt->scale), x = fRound(g_ipt->x), y = fRound(g_ipt->y);
	__shared__ float s_resX[109], s_resY[109], s_ang[109];

	// calculate haar responses for points within radius of 6*scale
	for (int index = threadIdx.x; index < 109; index += 42) {
		// Get X&Y offset of our sampling point (unscaled)
		int xOffset = dc_coord_x[index];
		int yOffset = dc_coord_y[index];
		float gauss = dc_gauss_lin[index];

		// Take the sample
		float haarXResult, haarYResult;
		haarXY(x+xOffset*s, y+yOffset*s, 2*s, &haarXResult, &haarYResult, gauss);

		// Store the sample and precomputed angle in shared memory
		s_resX[index] = haarXResult;
		s_resY[index] = haarYResult;
		s_ang[index] = getAngle(haarXResult, haarYResult);
	}

	__syncthreads(); // Wait until all thread finished taking their sample

	// calculate the dominant direction
	float sumX, sumY;
	float ang1, ang2, ang;
	float pi = M_PI;
	float pi_third = pi / 3.0f; // Size of the sliding window

	// Calculate ang1 at which this thread operates, 42 times at most
	ang1 = threadIdx.x * 0.15f;

	// Padded to 48 to allow efficient reduction by 24 threads without branching
	__shared__ float s_metrics[48];
	__shared__ float s_orientations[48];

	// Set the padding to 0, so it doesnt interfere.
	if (threadIdx.x < 6) {
		s_metrics[42 + threadIdx.x] = 0.0f;
	}

	// Each thread now computes one of the windows
	ang2 = ang1+pi_third > 2.0f*pi ? ang1-5.0f*pi_third : ang1+pi_third;
	sumX = sumY = 0.0f;

	// Find all the points that are inside the window
	// The x,y results computed above are now interpreted as points
	for (unsigned int k = 0; k < 109; k++) {
		ang = s_ang[k]; // Angle of vector to point

		// determine whether the point is within the window
		if (ang1 < ang2 && ang1 < ang && ang < ang2) {
			sumX += s_resX[k];
			sumY += s_resY[k];
		} else if (ang2 < ang1 &&
				((ang > 0.0f && ang < ang2) || (ang > ang1 && ang < 2.0f*pi) )) {
			sumX += s_resX[k];
			sumY += s_resY[k];
		}
	}

	// if the vector produced from this window is longer than all
	// previous vectors then this forms the new dominant direction
	s_metrics[threadIdx.x] = sumX*sumX + sumY*sumY;
	s_orientations[threadIdx.x] = getAngle(sumX, sumY);

	__syncthreads();

	/*
	 * The rest of this function finds the longest vector.
	 * The vector length is stored in metrics, while the
	 * corresponding orientation is stored in orientations
	 * with the same index.
	 */
#pragma unroll 4
	for (int threadCount = 24; threadCount >= 3; threadCount /= 2) {
		if (threadIdx.x < threadCount) {
			if (s_metrics[threadIdx.x] < s_metrics[threadIdx.x + threadCount]) {
				s_metrics[threadIdx.x] = s_metrics[threadIdx.x + threadCount];
				s_orientations[threadIdx.x] = s_orientations[threadIdx.x + threadCount];
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		float max = 0.0f, maxOrientation = 0.0f;
#pragma unroll 3
		for (int i = 0; i < 3; ++i) {
			if (s_metrics[i] > max) {
				max = s_metrics[i];
				maxOrientation = s_orientations[i];
			}
		}

		// assign orientation of the dominant response vector
		g_ipt->orientation = maxOrientation;
	}
}

#endif /* CUDA_DETECTIPOINTORIENTATIONSCUDA_CU */
