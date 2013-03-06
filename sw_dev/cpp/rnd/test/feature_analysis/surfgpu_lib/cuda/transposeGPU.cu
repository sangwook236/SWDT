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

#include <stdio.h>

#include "../transposeGPU.h"
#include "helper_funcs.h"

#include "transposeCUDA.cu"

void
transposeGPU(float *d_dst, size_t dst_pitch,
	float *d_src, size_t src_pitch,
	unsigned int width, unsigned int height)
{
	// execution configuration parameters
	dim3 threads(16, 16);
	dim3 grid(iDivUp(width, 16), iDivUp(height, 16));
	size_t shared_mem_size =
		(threads.x * threads.y + (threads.y - 1)) * sizeof(float);

	transposeCUDA<<<grid, threads, shared_mem_size>>>(
		d_dst, dst_pitch / sizeof(float),
		d_src, src_pitch / sizeof(float),
		width, height);
	CUDA_CHECK_MSG("transposeCUDA() execution failed");
}

