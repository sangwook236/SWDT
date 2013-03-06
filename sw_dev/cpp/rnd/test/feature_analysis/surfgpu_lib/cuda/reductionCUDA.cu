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

#ifndef CUDA_REDUCTIONCUDA_CU
#define CUDA_REDUCTIONCUDA_CU

#include "common_kernel.h"

#ifdef __DEVICE_EMULATION__
#	define EMUSYNC __syncthreads()
#else
#	define EMUSYNC
#endif

/* Bank-conflict free reduction
 * Based on the reduction example in the CUDA SDK
 */
__device__ void
reductionCUDA(float *data, unsigned int num_values, unsigned int tid)
{
	// do reduction in shared mem
	unsigned int stride = (unsigned int) ceilf(num_values / 2.0f);
	for (num_values >>= 1; num_values > 0;)
	{
		if (tid < num_values)
		{
			data[tid] += data[tid + stride];
		}
		__syncthreads();
		num_values = stride / 2;
		stride = (unsigned int) ceilf(stride / 2.0f);
	}
}

/* Optimized reduction of 64 values */
__device__ void
reduction64OptCUDA(volatile float *data, unsigned int tid)
{
	if (tid < 32)
	{
		data[tid] += data[tid + 32]; EMUSYNC;
		data[tid] += data[tid + 16]; EMUSYNC;
		data[tid] += data[tid +  8]; EMUSYNC;
		data[tid] += data[tid +  4]; EMUSYNC;
		data[tid] += data[tid +  2]; EMUSYNC;
		data[tid] += data[tid +  1]; EMUSYNC;
	}
}

#endif /* CUDA_REDUCTIONCUDA_CU */
