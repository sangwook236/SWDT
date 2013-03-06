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

#include <algorithm>
#include <iostream>
#include <vector>
#include <cstring>

#include <cuda_runtime.h>

#include "helper_funcs.h"
//--S [] 2013/03/06: Sang-Wook Lee
//#include "../ipoint.h"
#include "../ipointGPU.h"
//--E [] 2013/03/06: Sang-Wook Lee
#include "../matchSURFKeypointsGPU.h"

//--S [] 2013/03/06: Sang-Wook Lee
namespace surfgpu {
//--E [] 2013/03/06: Sang-Wook Lee

void linearizeDescriptors(float *dst, const IpVec &ipts);

void
getMatches(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches)
{
	// Allocate host memory
	float *lin_descs1 = new float[ipts1.size() * 64];
	float *lin_descs2 = new float[ipts2.size() * 64];
	int *indices = new int[ipts1.size()];
	float *dists = new float[ipts1.size()];

	linearizeDescriptors(lin_descs1, ipts1);
	linearizeDescriptors(lin_descs2, ipts2);

	// Allocate GPU memory
	float *d_lin_descs1;
	float *d_lin_descs2;
	int *d_indices;
	float *d_dists;
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_lin_descs1, ipts1.size() * 64 * sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_lin_descs2, ipts2.size() * 64 * sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_indices, ipts1.size() * sizeof(int)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_dists, ipts1.size() * sizeof(float)) );

	CUDA_SAFE_CALL( cudaMemcpy(d_lin_descs1, lin_descs1, ipts1.size() * 64 * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_lin_descs2, lin_descs2, ipts2.size() * 64 * sizeof(float), cudaMemcpyHostToDevice) );

	prepare_matchSURFKeypointsGPU(0.65f);
	matchSURFKeypointsGPU(d_indices, d_dists,
		d_lin_descs1, ipts1.size(), 64 * sizeof(float),
		d_lin_descs2, ipts2.size(), 64 * sizeof(float),
		64);

	CUDA_SAFE_CALL( cudaMemcpy(indices, d_indices, ipts1.size() * sizeof(int), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(dists, d_dists, ipts1.size() * sizeof(float), cudaMemcpyDeviceToHost) );

	for (size_t i = 0; i < ipts1.size(); i++)
	{
		if (indices[i] != -1)
		{
			surfgpu::Ipoint &ipt1 = ipts1[i];
			const surfgpu::Ipoint &match = ipts2[indices[i]];

			ipt1.dx = match.x - ipt1.x;
			ipt1.dy = match.y - ipt1.y;
			matches.push_back(std::make_pair(ipt1, match));
		}
	}

	delete[] lin_descs1;
	delete[] lin_descs2;
	delete[] indices;
	delete[] dists;

	CUDA_SAFE_CALL( cudaFree(d_lin_descs1) );
	CUDA_SAFE_CALL( cudaFree(d_lin_descs2) );
	CUDA_SAFE_CALL( cudaFree(d_indices) );
	CUDA_SAFE_CALL( cudaFree(d_dists) );
}

void
linearizeDescriptors(float *dst, const IpVec &ipts)
{
	for (size_t i = 0; i < ipts.size(); i++)
	{
		const surfgpu::Ipoint &p = ipts[i];
		std::memcpy(dst + i * 64, p.descriptor, 64 * sizeof(float));
	}
}

//--S [] 2013/03/06: Sang-Wook Lee
}  // namespace surfgpu
//--E [] 2013/03/06: Sang-Wook Lee
