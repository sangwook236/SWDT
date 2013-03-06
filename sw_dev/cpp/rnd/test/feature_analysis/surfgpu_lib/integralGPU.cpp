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

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cudpp.h>

//--S [] 2013/03/06: Sang-Wook Lee
//#include "integral.h"
#include "integralGPU.h"
//--E [] 2013/03/06: Sang-Wook Lee
#include "convertRGB2GrayFloatGPU.h"
#include "convertRGBA2GrayFloatGPU.h"
#include "transposeGPU.h"
#include "cuda/helper_funcs.h"
#include "cuda/cudpp_helper_funcs.h"

//--S [] 2013/03/06: Sang-Wook Lee
namespace surfgpu {
//--E [] 2013/03/06: Sang-Wook Lee

//! Computes the integral image of image img.  Assumes source image to be a
//! 32-bit floating point.  Returns cudaImage of 32-bit float form.
cudaImage* Integral(IplImage *src)
{
	assert(src->nChannels == 3 || src->nChannels == 4);
	assert(src->depth == IPL_DEPTH_8U);

	unsigned int *d_rgb_img;
	float *d_gray_img;
	float *d_int_img;
	float *d_int_img_tr;
	float *d_int_img_tr2;

	// Allocate device memory
	int img_width = src->width;
	int img_height = src->height;
	size_t rgb_img_pitch, gray_img_pitch, int_img_pitch, int_img_tr_pitch;
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_rgb_img, &rgb_img_pitch, img_width * sizeof(unsigned int), img_height) );
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_gray_img, &gray_img_pitch, img_width * sizeof(float), img_height) );
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_int_img, &int_img_pitch, img_width * sizeof(float), img_height) );
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_int_img_tr, &int_img_tr_pitch, img_height * sizeof(float), img_width) );
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_int_img_tr2, &int_img_tr_pitch, img_height * sizeof(float), img_width) );

	// Upload color image to GPU
	CUDA_SAFE_CALL(
			cudaMemcpy2D( d_rgb_img, rgb_img_pitch,
						  src->imageData, src->widthStep,
						  img_width * src->nChannels * src->depth / 8, img_height,
						  cudaMemcpyHostToDevice ) );

	if (src->nChannels == 3)
	{
		convertRGB2GrayFloatGPU(
			d_gray_img, gray_img_pitch,
			d_rgb_img, rgb_img_pitch,
			16, 8, 0,
			img_width, img_height);
	}
	else
	{
		convertRGBA2GrayFloatGPU(
			d_gray_img, gray_img_pitch,
			d_rgb_img, rgb_img_pitch,
			0x00ff0000, 0x0000ff00, 0x000000ff,
			16, 8, 0,
			img_width, img_height);
	}

	// Setup cudpp
	CUDPPHandle cudpp_lib;
	CUDPP_SAFE_CALL( cudppCreate(&cudpp_lib) );

	// Setup cudpp multiscan plans for computing the integral image
	CUDPPHandle mscan_plan, mscan_tr_plan;

	CUDPPConfiguration cudpp_conf;
	cudpp_conf.op = CUDPP_ADD;
	cudpp_conf.datatype = CUDPP_FLOAT;
	cudpp_conf.algorithm = CUDPP_SCAN;
	cudpp_conf.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

	CUDPP_SAFE_CALL( cudppPlan(cudpp_lib, &mscan_plan, cudpp_conf,
		img_width, img_height, gray_img_pitch / sizeof(float)) );

	CUDPP_SAFE_CALL( cudppPlan(cudpp_lib, &mscan_tr_plan, cudpp_conf,
		img_height, img_width, int_img_tr_pitch / sizeof(float)) );

	CUDPP_SAFE_CALL( cudppMultiScan(mscan_plan, d_int_img, d_gray_img, img_width, img_height) );
	transposeGPU(d_int_img_tr, int_img_tr_pitch,
		d_int_img, int_img_pitch,
		img_width, img_height);
	CUDPP_SAFE_CALL( cudppMultiScan(mscan_tr_plan, d_int_img_tr2, d_int_img_tr, img_height, img_width) );
	transposeGPU(d_int_img, int_img_pitch,
		d_int_img_tr2, int_img_tr_pitch,
		img_height, img_width);

	CUDPP_SAFE_CALL( cudppDestroyPlan(mscan_plan) );
	CUDPP_SAFE_CALL( cudppDestroyPlan(mscan_tr_plan) );
	CUDPP_SAFE_CALL( cudppDestroy( cudpp_lib ) );
	CUDA_SAFE_CALL( cudaFree(d_rgb_img) );
	CUDA_SAFE_CALL( cudaFree(d_gray_img) );
	CUDA_SAFE_CALL( cudaFree(d_int_img_tr) );
	CUDA_SAFE_CALL( cudaFree(d_int_img_tr2) );

	cudaImage *img = new cudaImage;
	img->width = img_width;
	img->height = img_height;
	img->data = reinterpret_cast<char*>(d_int_img);
	img->widthStep = int_img_pitch;

	return img;
}

//--S [] 2013/03/06: Sang-Wook Lee
}  // namespace surfgpu
//--E [] 2013/03/06: Sang-Wook Lee
