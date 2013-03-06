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
#include "../convertRGBA2GrayFloatGPU.h"

#include "convertRGBA2GrayFloatCUDA.cu"

void
convertRGBA2GrayFloatGPU(
	float *d_dst, size_t d_dst_pitch,
	const unsigned int *d_src, size_t d_src_pitch,
	unsigned int red_mask, unsigned int green_mask, unsigned int blue_mask,
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
	dim3 thread_block(min(img_width, 16), min(img_height, 16));
	dim3 block_grid(iDivUp(img_width, 16), iDivUp(img_height, 16));

	convertRGBA2GrayFloatCUDA<<<block_grid, thread_block>>>(
		d_dst, d_dst_pitch / sizeof(float),
		d_src, d_src_pitch / sizeof(unsigned int),
		red_mask, green_mask, blue_mask,
		red_shift, green_shift, blue_shift,
		img_width, img_height);
	CUDA_CHECK_MSG("convertRGBA2GrayFloatCUDA() execution failed\n");
}

