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

#ifndef CUDA_CONVERTRGB2GRAYFLOATCUDA_CU
#define CUDA_CONVERTRGB2GRAYFLOATCUDA_CU

#ifdef __DEVICE_EMULATION__
#	include <stdio.h>
#endif
#include "common_kernel.h"


/**	\brief Converts an RGB image to gray
 *	\param g_dst device pointer to gray data
 *	\param dst_pitch pitch used to access g_dst (in number of elements)
 *	\param g_src device pointer to RGB data
 *	\param src_pitch pitch used to access g_src (in number of elements)
 *	\param red_shift number of bits between red color channel and bit 0
 *	\param green_shift number of bits between green color channel and bit 0
 *	\param blue_shift number of bits between blue color channel and bit 0
 *	\param img_width_ints image width in number of 32-bit integers
 *	\param img_height image height
 *
 *	Execution configuration:
 *	  Thread block: { 16, 16 }
 *	  Block grid  : { ceil(img_width_in_ints / (block.x + 8)), ceil(img_height / block.y) }
 *	  Shared mem  : (thread_block.x + 8) * thread_block.y * sizeof(unsigned int)
 *
 *	The resulting gray image values are in the range [0.0, 1.0].
 *
 *	Each 16x16 thread block converts 32x16 RGB pixels to gray float.
 *	Shared memory is organized into 16 rows with 24 * 4 = 96 bytes per row.
 *
 *	Shared memory layout of a single row:
 *<pre>
 * int      0    1    2    3    4    5    6    7    8    9   10   11        22   23
 *       _____________________________________________________________     ___________
 * bytes |0123|0123|0123|0123|0123|0123|0123|0123|0123|0123|0123|0123|     |0123|0123|
 *       |----+----+----+----+----+----+----+----+----+----+----+----|     |----+----|
 * color |RGBR|GBRG|BRGB|RGBR|GBRG|BRGB|RGBR|GBRG|BRGB|RGBR|GBRG|BRGB| ... |GBRG|BRGB|
 *       |-----------------------------------------------------------|     |---------|
 * pixel | 0 1| 1 2| 2 3| 4 5| 5 6| 6 7| 8 9| 910|1011|1213|1314|1415|     |2930|3031|
 *       -------------------------------------------------------------     -----------
 *</pre>
 */
__global__ void
convertRGB2GrayFloatCUDA(
	float *g_dst, size_t dst_pitch,
	const unsigned int *g_src, size_t src_pitch,
	unsigned char red_shift, unsigned char green_shift, unsigned char blue_shift,
	unsigned int img_width_ints, unsigned int img_height)
{
	extern __shared__ unsigned int s_mem[];

	unsigned int x = IMUL(blockIdx.x, blockDim.x + 8) + threadIdx.x;
	unsigned int y = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (y >= img_height) return;

	unsigned int src_index = IMUL(y, src_pitch) + x;
	unsigned int pixel;
	if (x < img_width_ints)
	{
		pixel = g_src[src_index];
	}
	unsigned int smem_idx = IMUL(threadIdx.y, blockDim.x + 8) + threadIdx.x;
	s_mem[smem_idx] = pixel;
	__syncthreads();

	unsigned char *s_mem_uc = (unsigned char*)s_mem;

	unsigned int tid_px = IMUL(threadIdx.y, IMUL(blockDim.x + 8, 4)) + IMUL(threadIdx.x, 3);
	unsigned int red = s_mem_uc[tid_px + red_shift / 8];
	unsigned int green = s_mem_uc[tid_px + green_shift / 8];
	unsigned int blue = s_mem_uc[tid_px + blue_shift / 8];

	float gray = 0.3f * red + 0.59f * green + 0.11f * blue;
	unsigned int dst_index = IMUL(y, dst_pitch) + IMUL(blockIdx.x, blockDim.x * 2) + threadIdx.x;
	if (x < img_width_ints)
	{
		g_dst[dst_index] = gray / 255.0f;
	}

	/* Load 8*4 bytes from global memory so we can convert the remaining 16 RGB
	 * pixels.
	 */
	x += 8;
	if (threadIdx.x >= 8 && x < img_width_ints)
	{
		s_mem[smem_idx + 8] = g_src[src_index + 8];
	}
	__syncthreads();

	tid_px += IMUL(blockDim.x, 3);
	red = s_mem_uc[tid_px + red_shift / 8];
	green = s_mem_uc[tid_px + green_shift / 8];
	blue = s_mem_uc[tid_px + blue_shift / 8];

	gray = 0.3f * red + 0.59f * green + 0.11f * blue;
	dst_index += blockDim.x;
	if (x < img_width_ints)
	{
		g_dst[dst_index] = gray / 255.0f;
	}
}

#endif /* CUDA_CONVERTRGB2GRAYFLOATCUDA_CU */
