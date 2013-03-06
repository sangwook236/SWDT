/*
 * Copyright (C) 2009-2012 Andre Schulz, Florian Jung, Sebastian Hartte,
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

#include <assert.h>
#include <stdio.h>

#include "helper_funcs.h"
#include "../convertRGB2GrayFloatGPU.h"

#include "convertRGB2GrayFloatCUDA.cu"

void
convertRGB2GrayFloatGPU(
	float *d_dst, size_t d_dst_pitch,
	const unsigned int *d_src, size_t d_src_pitch,
	unsigned char red_shift, unsigned char green_shift, unsigned char blue_shift,
	unsigned int img_width, unsigned int img_height)
{
	assert(d_dst != 0);
	assert(d_dst_pitch > 0);
	assert(d_src != 0);
	assert(d_src_pitch > 0);
	assert(img_width > 0);
	assert(img_height > 0);

	// determine thread block and grid size
	unsigned int img_width_ints = ceilf(img_width * 3.0f / 4.0f);
	dim3 thread_block(min(img_width_ints, 16), min(img_height, 16));
	dim3 block_grid(iDivUp(img_width_ints, 16 + 8), iDivUp(img_height, 16));
	size_t smem_size = (thread_block.x + 8) * thread_block.y * sizeof(unsigned int);

	convertRGB2GrayFloatCUDA<<<block_grid, thread_block, smem_size>>>(
		d_dst, d_dst_pitch / sizeof(float),
		d_src, d_src_pitch / sizeof(unsigned int),
		red_shift, green_shift, blue_shift,
		img_width_ints, img_height);
	CUDA_CHECK_MSG("convertRGB2GrayFloatCUDA() execution failed\n");
}

