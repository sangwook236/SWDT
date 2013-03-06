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

#ifndef CUDA_NORMALIZESURFDESCRIPTORSCUDA_CU
#define CUDA_NORMALIZESURFDESCRIPTORSCUDA_CU

#ifdef __DEVICE_EMULATION
#	include <stdio.h>
#endif
#include "../surf_cudaipoint.h"
#include "common_kernel.h"

/**	\brief Normalize SURF descriptors
 *	\param g_ipoints device pointer to interest points and their descriptors
 *
 *	Execution configuration:
 *	  Thread block: { 64, 1, 1 }
 *	  Block grid  : { num_ipoints, 1 }
 *
 *	This kernel normalizes the feature vector by using 64 threads per block
 *	and one block per interest point.
 *	First, 4 of the threads sum up the lengths of the 16 sub-vectors (one for
 *	each sub-square). Then, one thread sums up the resulting 4 values and
 *	calculates the inverse square root of the value. The last step of
 *	normalization is performed by all 64 threads.
 */
__global__ void
normalizeSURFDescriptorsCUDA(surf_cudaIpoint* g_ipoints)
{
	surf_cudaIpoint* g_ipoint = g_ipoints + blockIdx.x;
	__shared__ float s_sums[4];

	if (threadIdx.x < 4) {
		float* g_lengths = g_ipoint->lengths[threadIdx.x];
		s_sums[threadIdx.x] = g_lengths[0] + g_lengths[1] + g_lengths[2] + g_lengths[3];
	}

	__syncthreads();

	float len = rsqrtf(s_sums[0] + s_sums[1] + s_sums[2] + s_sums[3]);

	g_ipoint->descriptor[threadIdx.x] *= len;
}

#endif /* CUDA_NORMALIZESURFDESCRIPTORSCUDA_CU */
