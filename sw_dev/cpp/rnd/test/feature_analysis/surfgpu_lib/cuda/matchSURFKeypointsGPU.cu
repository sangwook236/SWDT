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

#include "../defines.h"
#include "helper_funcs.h"
#include "../matchSURFKeypointsGPU.h"

#include "matchSURFKeypointsCUDA.cu"

void
prepare_matchSURFKeypointsGPU(float threshold)
{
	CUDA_SAFE_CALL( cudaMemcpyToSymbol("c_threshold", &threshold, sizeof(float)) );
}

void
matchSURFKeypointsGPU(
	int *d_result, float *d_dist,
	float *d_set1, size_t num_points_set1, size_t set1_pitch,
	float *d_set2, size_t num_points_set2, size_t set2_pitch,
	unsigned int desc_len)
{
	assert(d_result != 0);
	assert(d_dist != 0);
	assert(d_set1 != 0);
	assert(num_points_set1 > 0);
	assert(set1_pitch > 0);
	assert(d_set2 != 0);
	assert(num_points_set2 > 0);
	assert(set2_pitch > 0);
	assert(desc_len > 0);

	// execution config
	dim3 thread_block(32, 4, 1);
	dim3 block_grid(iDivUp(num_points_set1, thread_block.y), 1);

#if defined(SURF_MATCH_SIMPLE)
	matchSURFKeypointsCUDA<<<block_grid, thread_block>>>(
		d_result, d_dist,
		d_set1, num_points_set1, set1_pitch / sizeof(float),
		d_set2, num_points_set2, set2_pitch / sizeof(float));
	CUDA_CHECK_MSG("matchSURFKeypointsCUDA() execution failed\n");
#else
	matchSURFKeypoints2CUDA<<<block_grid, thread_block>>>(
		d_result, d_dist,
		d_set1, num_points_set1, set1_pitch / sizeof(float),
		d_set2, num_points_set2, set2_pitch / sizeof(float));
	CUDA_CHECK_MSG("matchSURFKeypoints2CUDA() execution failed\n");
#endif
}

