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

#include <cstring>

#include <cuda_runtime.h>

//--S [] 2013/03/06: Sang-Wook Lee
//#include "surf.h"
#include "surfGPU.h"
//--E [] 2013/03/06: Sang-Wook Lee
#include "cuda/helper_funcs.h"

#include "buildSURFDescriptorsGPU.h"
#include "detectIpointOrientationsGPU.h"
#include "normalizeSURFDescriptorsGPU.h"

//--S [] 2013/03/06: Sang-Wook Lee
namespace surfgpu {
//--E [] 2013/03/06: Sang-Wook Lee

//-------------------------------------------------------

//! Constructor
Surf::Surf(cudaImage *d_img, IpVec &ipts)
	: d_img(d_img), i_width(0), i_height(0), ipts(ipts), ca_intimg(NULL),
	  d_intimg_padded(NULL), d_ipoints(NULL)
{
	setIntImage(d_img);

	prepare_buildSURFDescriptorsGPU();
	prepare_detectIpointOrientationsGPU();
}

//! Destructor
Surf::~Surf()
{
	freeIntImage();
}
//-------------------------------------------------------

//! Describe all features in the supplied vector
void Surf::getDescriptors(bool upright)
{
	// Build the data structure used by the CUDA part
	surf_cudaIpoint *h_ipoints = new surf_cudaIpoint[ipts.size()];
	size_t ipoints_size = ipts.size() * sizeof(surf_cudaIpoint);
	std::memset(h_ipoints, 0, ipoints_size);
	for (size_t i = 0; i < ipts.size(); ++i) {
		const Ipoint &p = ipts[i];
		h_ipoints[i].x = p.x;
		h_ipoints[i].y = p.y;
		h_ipoints[i].scale = p.scale;
	}

	// And upload to the GPU
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_ipoints, ipoints_size) );
	CUDA_SAFE_CALL( cudaMemcpy(d_ipoints, h_ipoints, ipoints_size,
							  cudaMemcpyHostToDevice) );

	// Run the CUDA part
	computeDescriptors(upright);

	// Download interest points from the GPU
	CUDA_SAFE_CALL( cudaMemcpy(h_ipoints, d_ipoints, ipoints_size,
							  cudaMemcpyDeviceToHost) );

	// Read back orientation & descriptor from the CUDA part
	for (size_t i = 0; i < ipts.size(); ++i) {
		Ipoint &p = ipts[i];
		p.orientation = h_ipoints[i].orientation;
		std::memcpy(p.descriptor, h_ipoints[i].descriptor, 64 * sizeof(float));
	}

	CUDA_SAFE_CALL( cudaFree(d_ipoints) );
	delete[] h_ipoints;
}

//! Set or re-set the integral image source
void Surf::setIntImage(cudaImage *d_img)
{
	// Change the source image
	this->d_img = d_img;

	unsigned int width = d_img->width;
	unsigned int height = d_img->height;
	unsigned int padded_width = width + 1;
	unsigned int padded_height = height + 1;
	unsigned int padded_rowsize = padded_width * sizeof(float);
	unsigned int padded_size = padded_rowsize * padded_height;

	// Redefine width, height only if image has changed size
	bool img_size_changed = d_img->width != this->i_width
							|| d_img->height != this->i_height;
	if (img_size_changed)
	{
		// Reallocate GPU memory
		freeIntImage();

		this->i_width = d_img->width;
		this->i_height = d_img->height;

		// Allocate a channel that has 32 bits in the x-component (A single float value)
		cudaChannelFormatDesc channelDesc;
		channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		CUDA_SAFE_CALL( cudaMallocArray(&ca_intimg, &channelDesc,
									   padded_width, padded_height) );

		CUDA_SAFE_CALL( cudaMalloc((void**)&d_intimg_padded, padded_size) );
	}

	CUDA_SAFE_CALL( cudaMemset(d_intimg_padded, 0, padded_size) );
	CUDA_SAFE_CALL( cudaMemcpy2D(d_intimg_padded + padded_width + 1, padded_rowsize,
								d_img->data, d_img->widthStep,
								width * sizeof(float), height,
								cudaMemcpyDeviceToDevice) );

	CUDA_SAFE_CALL( cudaMemcpy2DToArray(
							ca_intimg, 0, 0,
							d_intimg_padded, padded_rowsize,
							padded_rowsize, padded_height,
							cudaMemcpyDeviceToDevice) );
}

//! Free integral image
void Surf::freeIntImage()
{
	CUDA_SAFE_CALL( cudaFree(d_intimg_padded) );
	CUDA_SAFE_CALL( cudaFreeArray(ca_intimg) );
}

void
Surf::setIpoints(IpVec &ipts)
{
	this->ipts = ipts;
}

void
Surf::computeDescriptors(int upright)
{
	if (!upright) {
		prepare2_detectIpointOrientationsGPU(ca_intimg);
		detectIpointOrientationsGPU(d_ipoints, ipts.size());
	}

	prepare2_buildSURFDescriptorsGPU(ca_intimg);
	buildSURFDescriptorsGPU(d_ipoints, upright, ipts.size());

	normalizeSURFDescriptorsGPU(d_ipoints, ipts.size());
}

//--S [] 2013/03/06: Sang-Wook Lee
}  // namespace surfgpu
//--E [] 2013/03/06: Sang-Wook Lee
