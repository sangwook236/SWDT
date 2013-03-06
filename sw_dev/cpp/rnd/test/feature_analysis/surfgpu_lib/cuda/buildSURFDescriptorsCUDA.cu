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

#ifndef CUDA_BUILDSURFDESCRIPTORSCUDA_CU
#define CUDA_BUILDSURFDESCRIPTORSCUDA_CU

#ifdef __DEVICE_EMULATION__
#	include <stdio.h>
#endif
#include "../surf_cudaipoint.h"
#include "common_kernel.h"

// Texture reference to the integral image; needed by haarXY()
texture<float, 2, cudaReadModeElementType> integralImage;
#include "haarXY.cu"

__constant__ float dc_gauss33[12][12];

/**	\brief Build SURF descriptor for an interest point
 *	\param g_ipoints device pointer to interest points
 *	\param upright compute upright SURF descriptor or not
 *
 *  This kernel builds the descriptors for an interest point.
 *  Execution configuration:
 *	  Thread block: { 5, 5, 16 }
 *	  Block grid  : { num_ipoints, 1 }
 *
 *  Overview:
 *  Each thread takes a haar X/Y sample and stores it in
 *  shared memory (rx,ry).
 *  For each subsquare (threadIdx.z is the subSquare id),
 *  the four sums (dx,dy,|dx|,|dy|) are built and stored in
 *  global memory (desc, outDesc).
 *  Then, one thread per sub square computes the squared
 *  length of a sub-square and stores it in global memory.
 */
__global__ void
buildSURFDescriptorsCUDA(surf_cudaIpoint* g_ipoints, int upright)
{
	const int iPointIndex = blockIdx.x; // The index in the one-dimensional ipoint array
	const int samplePointXIndex = threadIdx.x; // The x-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int samplePointYIndex = threadIdx.y; // The y-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int subSquareId = threadIdx.z; // The index of the sub-square
	const int subSquareX = (subSquareId % 4); // X-Index of the sub-square
	const int subSquareY = (subSquareId / 4); // Y-Index of the sub-square

	surf_cudaIpoint *g_ipt = g_ipoints + iPointIndex; // Pointer to the interest point processed by the current block
	int x = fRound(g_ipt->x);
	int y = fRound(g_ipt->y);
	float scale = g_ipt->scale;

	float * const g_desc = g_ipt->descriptor; // Pointer to the interest point descriptor
	float co, si; // Precomputed cos&sin values for the rotation of this interest point

	if (!upright) {
		co = cosf(g_ipt->orientation);
		si = sinf(g_ipt->orientation);
	}

	int roundedScale = fRound(scale);

	// Calculate the relative (to x,y) coordinate of sampling point
	int sampleXOffset = subSquareX * 5 - 10 + samplePointXIndex;
	int sampleYOffset = subSquareY * 5 - 10 + samplePointYIndex;

	// Get Gaussian weighted x and y responses
	float gauss = dc_gauss33[abs(sampleYOffset)][abs(sampleXOffset)];

	// Get absolute coords of sample point on the rotated axis
	int sampleX, sampleY;

	if (!upright) {
		sampleX = fRound(x + (-sampleXOffset*scale*si + sampleYOffset*scale*co));
		sampleY = fRound(y + ( sampleXOffset*scale*co + sampleYOffset*scale*si));
	} else {
		sampleX = fRound(x + sampleXOffset*scale);
		sampleY = fRound(y + sampleYOffset*scale);
	}

	// Take the sample (Haar wavelet response in x&y direction)
	float xResponse, yResponse;
	haarXY(sampleX, sampleY, roundedScale, &xResponse, &yResponse, gauss);

	// Calculate ALL x+y responses for the interest point in parallel
	__shared__ float s_rx[16][5][5];
	__shared__ float s_ry[16][5][5];

	if (!upright) {
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex] = -xResponse*si + yResponse*co;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex] = xResponse*co + yResponse*si;
	} else {
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex] = xResponse;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex] = yResponse;
	}

	// TODO: Can this be optimized? It waits for the results of ALL 400 threads, although they are
	// independent in blocks of 25! (Further work)
	__syncthreads(); // Wait until all 400 threads have written their results

	__shared__ float s_sums[16][4][5]; // For each sub-square, for the four values (dx,dy,|dx|,|dy|), this contains the sum over five values.
	__shared__ float s_outDesc[16][4]; // The output descriptor partitioned into 16 bins (one for each subsquare)

	// Only five threads per sub-square sum up five values each
	if (threadIdx.y == 0) {
		// Temporary sums
		float tdx = 0.0f, tdy = 0.0f, tmdx = 0.0f, tmdy = 0.0f;

		for (int sy = 0; sy < 5; ++sy) {
			tdx += s_rx[subSquareId][threadIdx.x][sy];
			tdy += s_ry[subSquareId][threadIdx.x][sy];
			tmdx += fabsf(s_rx[subSquareId][threadIdx.x][sy]);
			tmdy += fabsf(s_ry[subSquareId][threadIdx.x][sy]);
		}

		// Write out the four sums to the shared memory
		s_sums[subSquareId][0][threadIdx.x] = tdx;
		s_sums[subSquareId][1][threadIdx.x] = tdy;
		s_sums[subSquareId][2][threadIdx.x] = tmdx;
		s_sums[subSquareId][3][threadIdx.x] = tmdy;
	}

	__syncthreads(); // Wait until all threads have summed their values

	// Only four threads per sub-square can now write out the descriptor
	if (threadIdx.x < 4 && threadIdx.y == 0) {
		const float* s_src = s_sums[subSquareId][threadIdx.x]; // Pointer to the sum this thread will write out
		float out = s_src[0] + s_src[1] + s_src[2] + s_src[3] + s_src[4]; // Build the last sum for the value this thread writes out
		int subSquareOffset = (subSquareX + subSquareY * 4) * 4; // Calculate the offset in the descriptor for this sub-square
		g_desc[subSquareOffset + threadIdx.x] = out; // Write the value to the descriptor
		s_outDesc[subSquareId][threadIdx.x] = out; // Write the result to shared memory too, this will be used by the last thread to precompute parts of the length
	}

	__syncthreads();

	// One thread per sub-square now computes the length of the description vector for a sub-square and writes it to global memory
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		g_ipt->lengths[subSquareX][subSquareY] = s_outDesc[subSquareId][0] * s_outDesc[subSquareId][0]
			+ s_outDesc[subSquareId][1] * s_outDesc[subSquareId][1]
			+ s_outDesc[subSquareId][2] * s_outDesc[subSquareId][2]
			+ s_outDesc[subSquareId][3] * s_outDesc[subSquareId][3];
	}
}

#endif /* CUDA_BUILDSURFDESCRIPTORSCUDA_CU */
