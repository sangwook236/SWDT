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
#include "../normalizeSURFDescriptorsGPU.h"

#include "normalizeSURFDescriptorsCUDA.cu"

void
normalizeSURFDescriptorsGPU(surf_cudaIpoint *d_ipoints, size_t num_ipoints)
{
	assert(d_ipoints != 0);
	assert(num_ipoints > 0);

	dim3 thread_block(64, 1, 1);
	dim3 block_grid(num_ipoints, 1);

	normalizeSURFDescriptorsCUDA<<<block_grid, thread_block>>>(d_ipoints);
	CUDA_CHECK_MSG("normalizeSURFDescriptorsCUDA() execution failed");
}

