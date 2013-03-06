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
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "helper_funcs.h"
#include "../fasthessian_cudaipoint.h"
#include "../cudaimage.h"
#include "../nonMaxGPU.h"

#include "nonMaxCUDA.cu"

/////////////////////////////////////////////////////////
// Definitions and constants
/////////////////////////////////////////////////////////

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 8

//-------------------------------------------------------

void
prepare_nonMaxGPU(const int *h_lobe_map, size_t size)
{
	CUDA_SAFE_CALL( cudaMemcpyToSymbol("dc_lobe_map", h_lobe_map, size) );
}

//-------------------------------------------------------

void
nonMaxGPU(float *d_det, unsigned int det_width, unsigned int det_height,
		  size_t det_width_step, fasthessian_cudaIpoint *d_points,
		  const int *border_cache, int octaves, int intervals, int init_sample,
		  float thres)
{
	int *atomicCounter_ptr;
	size_t atomicCounter_size;
	CUDA_SAFE_CALL( cudaGetSymbolAddress((void**)&atomicCounter_ptr, "atomicCounter") );
	CUDA_SAFE_CALL( cudaGetSymbolSize(&atomicCounter_size, "atomicCounter") );
	CUDA_SAFE_CALL( cudaMemset(atomicCounter_ptr, 0, atomicCounter_size) );

	for (int o = 0; o < octaves; o++) {
		// For each octave double the sampling step of the previous
		int step = init_sample * (1 << o);
		int border = border_cache[o];

		// Calculate grid size
		int steps_x = (det_width - 2 * border + 3 * step - 1) / (3 * step);
		int steps_y = (det_height - 2 * border + 3 * step - 1) / (3 * step);
		int steps_i = (intervals - 2) / 3 + 1;

		dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 grid((steps_x + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (steps_y + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
		grid.x *= steps_i; /* Calculate all intervals with one single kernel call */

		// Launch kernel
#ifdef DEBUG
		printf("Call NMS kernel: octave %d, steps %dx%dx%d, border %d, grid dim %dx%d, block dim %dx%d\n", o, steps_x, steps_y, steps_i, border, grid.x, grid.y, block.x, block.y);
#endif

		nonMaxCUDA<<<grid, block>>>(d_det, det_width, det_height,
			det_width_step, d_points,
			intervals, o, step, steps_i, border, thres, init_sample);
		CUDA_CHECK_MSG("nonMaxCUDA() execution failed");
	}
}

