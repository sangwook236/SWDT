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

#ifndef CUDA_TRANSPOSECUDA_CU
#define CUDA_TRANSPOSECUDA_CU

#ifdef __DEVICE_EMULATION__
#	include <stdio.h>
#endif

#include "common_kernel.h"

/** \brief Transpose a matrix of values
 *	\param g_dst device pointer to write results to
 *	\param s_dst_pitch number of elements per row in g_dst
 *	\param g_src device pointer to source matrix
 *	\param s_src_pitch number of elements per row in g_src
 *	\param img_width image width
 *	\param img_height image height
 *
 *	Recommended execution configuration:
 *	  Thread block: { 16, 16 }
 *	  Block grid  : { ceil(img_width / block.x), ceil(img_height / block.y) }
 */
// TODO: resolve global memory partition camping (see transposeNew SDK example)
__global__ void
transposeCUDA(
	float *g_dst, size_t s_dst_pitch,
	const float *g_src, size_t s_src_pitch,
	unsigned int img_width, unsigned int img_height)
{
	extern __shared__ float s_mem[];
	unsigned int x = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int y = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	unsigned int src_offset = IMUL(y, s_src_pitch) + x;
	unsigned int smem_offset = IMUL(threadIdx.y, blockDim.x) + threadIdx.x
		+ threadIdx.y;

	// Load data into shared memory
	if (y < img_height)
	{
		s_mem[smem_offset] = g_src[src_offset];
	}

	__syncthreads();

	// Compute smem_offset so that we read the values transposed
	smem_offset = IMUL(threadIdx.x, blockDim.x) + threadIdx.y + threadIdx.x;

	// Compute destination offset
	x = IMUL(blockIdx.y, blockDim.x) + threadIdx.x;
	y = IMUL(blockIdx.x, blockDim.y) + threadIdx.y;
	unsigned int dst_offset = IMUL(y, s_dst_pitch) + x;

	// Write data back to global memory
	if (y < img_width)
	{
		g_dst[dst_offset] = s_mem[smem_offset];
	}
}

#endif /* CUDA_TRANSPOSECUDA_CU */
