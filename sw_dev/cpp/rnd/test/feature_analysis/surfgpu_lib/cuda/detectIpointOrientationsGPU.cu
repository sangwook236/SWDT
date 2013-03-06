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

#include <cuda_runtime.h>

#include "helper_funcs.h"
#include "../surf_cudaipoint.h"
#include "../detectIpointOrientationsGPU.h"

#include "detectIpointOrientationsCUDA.cu"

// The following table contains the x coordinate LUT for the orientation detection threads
static const int coord_x[109] = {
	-5, -5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
	-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,
	 0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
	 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5
};

// The following table contains the y coordinate LUT for the orientation detection threads
static const int coord_y[109] = {
	-3, -2, -1,  0,  1,  2,  3, -4, -3, -2, -1,  0,  1,  2,  3,  4, -5, -4, -3, -2, -1, 0, 1,
	 2,  3,  4,  5, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1,  0, 1, 2,
	 3,  4,  5, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1,  0,  1, 2, 3, 4,
	 5, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1,  0,  1,  2,  3, 4, 5,
	-4, -3, -2, -1,  0,  1,  2,  3,  4, -3, -2, -1,  0,  1,  2,  3
};

// The following table contains the gauss coordinate LUT for the orientation detection threads
// This was originally a 2 dimensional array, but for efficiency reasons and since the orientation detection
// runs in a one-dimensional block, this array has been made one-dimensional and corresponds to the
// x and y lookup tables above
static const float gauss_lin[109] = {
	0.000958195f, 0.00167749f, 0.00250251f, 0.00318132f, 0.00250251f, 0.00167749f, 0.000958195f, 0.000958195f, 0.00196855f, 0.00344628f, 0.00514125f, 0.00653581f, 0.00514125f, 0.00344628f, 0.00196855f, 0.000958195f, 0.000695792f, 0.00167749f,
	0.00344628f, 0.00603331f, 0.00900064f, 0.0114421f, 0.00900064f, 0.00603331f, 0.00344628f, 0.00167749f, 0.000695792f, 0.001038f, 0.00250251f, 0.00514125f, 0.00900064f, 0.0134274f, 0.0170695f, 0.0134274f, 0.00900064f, 0.00514125f, 0.00250251f,
	0.001038f, 0.00131956f, 0.00318132f, 0.00653581f, 0.0114421f, 0.0170695f, 0.0216996f, 0.0170695f, 0.0114421f, 0.00653581f, 0.00318132f, 0.00131956f, 0.00142946f, 0.00344628f, 0.00708015f, 0.012395f, 0.0184912f, 0.0235069f, 0.0184912f, 0.012395f,
	0.00708015f, 0.00344628f, 0.00142946f, 0.00131956f, 0.00318132f, 0.00653581f, 0.0114421f, 0.0170695f, 0.0216996f, 0.0170695f, 0.0114421f, 0.00653581f, 0.00318132f, 0.00131956f, 0.001038f, 0.00250251f, 0.00514125f, 0.00900064f, 0.0134274f,
	0.0170695f, 0.0134274f, 0.00900064f, 0.00514125f, 0.00250251f, 0.001038f, 0.000695792f, 0.00167749f, 0.00344628f, 0.00603331f, 0.00900064f, 0.0114421f, 0.00900064f, 0.00603331f, 0.00344628f, 0.00167749f, 0.000695792f, 0.000958195f, 0.00196855f,
	0.00344628f, 0.00514125f, 0.00653581f, 0.00514125f, 0.00344628f, 0.00196855f, 0.000958195f, 0.000958195f, 0.00167749f, 0.00250251f, 0.00318132f, 0.00250251f, 0.00167749f, 0.000958195f
};

void
prepare_detectIpointOrientationsGPU()
{
	CUDA_SAFE_CALL( cudaMemcpyToSymbol("dc_gauss_lin", gauss_lin, sizeof(gauss_lin)) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol("dc_coord_x", coord_x, sizeof(coord_x)) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol("dc_coord_y", coord_y, sizeof(coord_y)) );
}

void
prepare2_detectIpointOrientationsGPU(cudaArray *ca_intimg)
{
	// integralImage refers to the texture reference from
	// detectIpointOrientiationsCUDA.cu which is included above.
	integralImage.filterMode = cudaFilterModePoint; // We don't use interpolation
	integralImage.normalized = false; // Don't normalize texture coordinates
	/* Clamping saves us some boundary checks */
	integralImage.addressMode[0] = cudaAddressModeClamp;
	integralImage.addressMode[1] = cudaAddressModeClamp;
	integralImage.addressMode[2] = cudaAddressModeClamp;

	CUDA_SAFE_CALL( cudaBindTextureToArray(integralImage, ca_intimg) );
}

void
detectIpointOrientationsGPU(
	surf_cudaIpoint *d_ipoints, size_t num_ipoints)
{
	assert(d_ipoints != 0);
	assert(num_ipoints > 0);

	dim3 thread_block(42, 1, 1);
	dim3 block_grid(num_ipoints, 1);

	detectIpointOrientiationsCUDA<<<block_grid, thread_block>>>(d_ipoints);
	CUDA_CHECK_MSG("detectIpointOrientationsCUDA() execution failed");
}

