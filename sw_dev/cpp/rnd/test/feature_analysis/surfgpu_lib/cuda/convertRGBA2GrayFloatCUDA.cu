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

#ifndef CUDA_CONVERTRGBA2GRAYFLOATCUDA_CU
#define CUDA_CONVERTRGBA2GRAYFLOATCUDA_CU

#ifdef __DEVICE_EMULATION__
#	include <stdio.h>
#endif
#include "common_kernel.h"


/**	\brief Converts an RGBA image to gray
 *	\param g_dst device pointer to gray data
 *	\param dst_pitch pitch used to access g_dst (in number of elements)
 *	\param g_src device pointer to RGBA data
 *	\param src_pitch pitch used to access g_src (in number of elements)
 *	\param red_mask bit mask used to isolate the red color channel of a pixel
 *	\param green_mask bit mask used to isolate the green color channel of a pixel
 *	\param blue_mask bit mask used to isolate the blue color channel of a pixel
 *	\param red_shift number of bits between red color channel and bit 0
 *	\param green_shift number of bits between green color channel and bit 0
 *	\param blue_shift number of bits between blue color channel and bit 0
 *	\param img_width image width
 *	\param img_height image height
 *
 *	Execution configuration:
 *	  Thread block: { 16, 16 }
 *	  Block grid  : { ceil(img_width / block.x), ceil(img_height / block.y) }
 *
 *	This function expects that both g_dst and g_src have been allocated by
 *	cudaMallocPitch(), such that the OOB check in X can be omitted because
 *	of the padding of each row.
 *	The resulting gray image values are in the range [0.0, 1.0].
 */
__global__ void
convertRGBA2GrayFloatCUDA(
	float *g_dst, size_t dst_pitch,
	const unsigned int *g_src, size_t src_pitch,
	unsigned int red_mask, unsigned int green_mask, unsigned int blue_mask,
	unsigned char red_shift, unsigned char green_shift, unsigned char blue_shift,
	unsigned int img_width, unsigned int img_height)
{
	unsigned int x = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int y = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	unsigned int index = IMUL(y, src_pitch) + x;

	if (y >= img_height) return;

	unsigned int pixel = g_src[index];

	unsigned int red = (pixel & red_mask) >> red_shift;
	unsigned int green = (pixel & green_mask) >> green_shift;
	unsigned int blue = (pixel & blue_mask) >> blue_shift;

	float gray = 0.3f * red + 0.59f * green + 0.11f * blue;
	index = IMUL(y, dst_pitch) + x;
	g_dst[index] = gray / 255.0f;
}

#endif /* CUDA_CONVERTRGBA2GRAYFLOATCUDA_CU */
