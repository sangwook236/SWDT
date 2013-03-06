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
#include "../buildSURFDescriptorsGPU.h"

#include "buildSURFDescriptorsCUDA.cu"

/*
 * The following table is padded to 12x12 (up from 11x11) so it can be loaded
 * efficiently by the 4x4 threads.
 */
static const float gauss33[12][12] = {
	0.014614763f,0.013958917f,0.012162744f,0.00966788f,0.00701053f,0.004637568f,0.002798657f,0.001540738f,0.000773799f,0.000354525f,0.000148179f,0.0f,
	0.013958917f,0.013332502f,0.011616933f,0.009234028f,0.006695928f,0.004429455f,0.002673066f,0.001471597f,0.000739074f,0.000338616f,0.000141529f,0.0f,
	0.012162744f,0.011616933f,0.010122116f,0.008045833f,0.005834325f,0.003859491f,0.002329107f,0.001282238f,0.000643973f,0.000295044f,0.000123318f,0.0f,
	0.00966788f,0.009234028f,0.008045833f,0.006395444f,0.004637568f,0.003067819f,0.001851353f,0.001019221f,0.000511879f,0.000234524f,9.80224E-05f,0.0f,
	0.00701053f,0.006695928f,0.005834325f,0.004637568f,0.003362869f,0.002224587f,0.001342483f,0.000739074f,0.000371182f,0.000170062f,7.10796E-05f,0.0f,
	0.004637568f,0.004429455f,0.003859491f,0.003067819f,0.002224587f,0.001471597f,0.000888072f,0.000488908f,0.000245542f,0.000112498f,4.70202E-05f,0.0f,
	0.002798657f,0.002673066f,0.002329107f,0.001851353f,0.001342483f,0.000888072f,0.000535929f,0.000295044f,0.000148179f,6.78899E-05f,2.83755E-05f,0.0f,
	0.001540738f,0.001471597f,0.001282238f,0.001019221f,0.000739074f,0.000488908f,0.000295044f,0.00016243f,8.15765E-05f,3.73753E-05f,1.56215E-05f,0.0f,
	0.000773799f,0.000739074f,0.000643973f,0.000511879f,0.000371182f,0.000245542f,0.000148179f,8.15765E-05f,4.09698E-05f,1.87708E-05f,7.84553E-06f,0.0f,
	0.000354525f,0.000338616f,0.000295044f,0.000234524f,0.000170062f,0.000112498f,6.78899E-05f,3.73753E-05f,1.87708E-05f,8.60008E-06f,3.59452E-06f,0.0f,
	0.000148179f,0.000141529f,0.000123318f,9.80224E-05f,7.10796E-05f,4.70202E-05f,2.83755E-05f,1.56215E-05f,7.84553E-06f,3.59452E-06f,1.50238E-06f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f
};

void
prepare_buildSURFDescriptorsGPU()
{
	CUDA_SAFE_CALL( cudaMemcpyToSymbol("dc_gauss33", gauss33, sizeof(gauss33)) );
}

void
prepare2_buildSURFDescriptorsGPU(cudaArray *ca_intimg)
{
	integralImage.filterMode = cudaFilterModePoint; // We don't use interpolation
	integralImage.normalized = false; // Don't normalize texture coordinates
	/* Clamping saves us some boundary checks */
	integralImage.addressMode[0] = cudaAddressModeClamp;
	integralImage.addressMode[1] = cudaAddressModeClamp;
	integralImage.addressMode[2] = cudaAddressModeClamp;

	CUDA_SAFE_CALL( cudaBindTextureToArray(integralImage, ca_intimg) );
}

void
buildSURFDescriptorsGPU(surf_cudaIpoint *d_ipoints, int upright,
	size_t num_ipoints)
{
	assert(d_ipoints != 0);
	assert(num_ipoints > 0);

	dim3 thread_block(5, 5, 16);
	dim3 block_grid(num_ipoints, 1);

	buildSURFDescriptorsCUDA<<<block_grid, thread_block>>>(d_ipoints, upright);
	CUDA_CHECK_MSG("buildSURFDescriptorsCUDA() execution failed");
}

