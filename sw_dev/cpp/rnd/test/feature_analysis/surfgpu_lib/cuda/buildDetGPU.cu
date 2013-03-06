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
#include "../cudaimage.h"
#include "../buildDetGPU.h"

#include "buildDetCUDA.cu"

/////////////////////////////////////////////////////////
// Definitions and constants
/////////////////////////////////////////////////////////

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 8

/////////////////////////////////////////////////////////
// Kernel launcher functions (Host)
/////////////////////////////////////////////////////////

void
prepare_buildDetGPU(const int *lobe_cache_unique, size_t size)
{
	CUDA_SAFE_CALL( cudaMemcpyToSymbol("dc_lobe_cache_unique", lobe_cache_unique, size) );
}

void
buildDetGPU(cudaImage *d_img, float *d_det,	const int *border_cache,
	int octaves, int intervals, int init_sample)
{
	// Calculate step size
	int step = init_sample;

	// Get border size
	int border = border_cache[0];

	// Calculate grid size
	int steps_x = (d_img->width - 2 * border + step - 1) / step;
	int steps_y = (d_img->height - 2 * border + step - 1) / step;
	dim3 block(16, 4, 6);
	dim3 grid((steps_x + block.x - 1) / block.x,
			  (steps_y + block.y - 1) / block.y);

	// Launch kernel
#ifdef DEBUG
	printf("Call determinant kernel: octave %d, steps %dx%d, border %d, grid dim %dx%d, block dim %dx%dx%d\n",
			0, steps_x, steps_y, border, grid.x, grid.y, block.x, block.y, block.z);
#endif
	buildDetCUDA_smem_bf<<<grid, block>>>((float *)d_img->data, d_det,
		d_img->width, d_img->height, d_img->widthStep / sizeof(float),
		intervals, 0, step, border);
	CUDA_CHECK_MSG("buildDetCUDA_smem_bf() execution failed");

	// For octaves > 0, we only compute the higher 2 intervals.
	intervals = 2;
	for (int o = 1; o < octaves; o++) {
		// Calculate step size
		step = init_sample * (1 << o);

		// Get border size
		border = border_cache[o];

		// Calculate grid size
		steps_x = (d_img->width - 2 * border + step - 1) / step;
		steps_y = (d_img->height - 2 * border + step - 1) / step;
		dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 grid((steps_x + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (steps_y + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
		grid.x *= intervals;

		// Launch kernel
#ifdef DEBUG
		printf("Call determinant kernel: octave %d, steps %dx%d, border %d, grid dim %dx%d, block dim %dx%d\n",
				o, steps_x, steps_y, border, grid.x, grid.y, block.x, block.y);
#endif
		buildDetCUDA<<<grid, block>>>((float *)d_img->data, d_det,
			d_img->width, d_img->height, d_img->widthStep / sizeof(float),
			intervals, o, step, border);
		CUDA_CHECK_MSG("buildDetCUDA() execution failed");
	}
}

