/*
 * Copyright (C) 2009-2010 Andre Schulz, Florian Jung, Sebastian Hartte,
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

#ifndef CUDA_BUILDDETCUDA_CU
#define CUDA_BUILDDETCUDA_CU

#ifdef __DEVICE_EMULATION__
#	include <stdio.h>
#endif

/////////////////////////////////////////////////////////
// Definitions and constants
/////////////////////////////////////////////////////////

__constant__ int dc_lobe_cache_unique[10];

__device__ float BoxIntegral(float *data, int width, int height, size_t widthStep,
							 int row, int col, int rows, int cols);

/////////////////////////////////////////////////////////
// Kernel Code (Device)
/////////////////////////////////////////////////////////

/** \brief Compute determinants
 *	\param g_img device pointer to integral image
 *	\param g_det device pointer to save resulting determinants to
 *	\param i_width integral image width
 *	\param i_height integral image height
 *	\param i_widthStep number of elements in a row of the integral image
 *	\param intervals number of intervals to compute
 *	\param o octave to compute
 *	\param step step size in X/Y direction in pixels
 *	\param border number of border pixels
 *
 *	Computation is done pixel-wise where one thread computes the determinant of
 *	one pixel.
 *
 *	Recommended execution configuration:
 *	  Thread block: { 16, 8 }
 *	  Block grid  : { ceil(i_width / block.x), ceil(i_height, block.y) }
 */
__global__ void
buildDetCUDA(float *g_img, float *g_det,
			 int i_width, int i_height, size_t i_widthStep,
			 int intervals, int o, int step, int border)
{
	// Get current interval
	const int interval_size = gridDim.x / intervals;
	// For octaves > 0, we only compute the higher 2 intervals.
	const int i = blockIdx.x / interval_size + (o > 0) * 2;

	// Get current column and row
	const int c = ((((blockIdx.x % interval_size) * blockDim.x) + threadIdx.x) * step) + border;
	const int r = ((blockIdx.y * blockDim.y + threadIdx.y) * step) + border;

	if (c >= i_width - border || r >= i_height - border)
		return;

	// Construct filter
	const int l = dc_lobe_cache_unique[o * intervals + i];
	const int w = 3 * l;
	const int b = w / 2;

	// Caluclate box integrals
	float Dxx = BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l + 1, c - b, 2 * l - 1, w)
			  - BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l + 1, c - l / 2, 2 * l - 1, l) * 3.0f;
	float Dyy = BoxIntegral(g_img, i_width, i_height, i_widthStep, r - b, c - l + 1, w, 2 * l - 1)
			  - BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l / 2, c - l + 1, l, 2 * l - 1) * 3.0f;
	float Dxy = BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l, c + 1, l, l)
			  + BoxIntegral(g_img, i_width, i_height, i_widthStep, r + 1, c - l, l, l)
			  - BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l, c - l, l, l)
			  - BoxIntegral(g_img, i_width, i_height, i_widthStep, r + 1, c + 1, l, l);

	// Normalise the filter responses with respect to their size
	float inverse_area = 1.0f / (w * w);
	Dxx *= inverse_area;
	Dyy *= inverse_area;
	Dxy *= inverse_area;

	// Get the sign of the laplacian
	const float lap_sign = (Dxx + Dyy >= 0.0f ? 1.0f : -1.0f);

	// Get the determinant of hessian response
	float determinant = (Dxx * Dyy - 0.81f * Dxy * Dxy);
	unsigned int cur_intvl = o * intervals + i;
	unsigned int save_idx = cur_intvl * i_width * i_height + (r * i_width + c);
	g_det[save_idx] = determinant < 0.0f ? 0.0f : lap_sign * determinant;
}

//-------------------------------------------------------

/**	\brief Compute determinants using shared memory
 *	\param g_img device pointer to integral image
 *	\param g_det device pointer to save resulting determinants to
 *	\param i_width integral image width
 *	\param i_height integral image height
 *	\param i_widthStep number of elements in a row of the integral image
 *	\param intervals number of intervals to compute
 *	\param o octave to compute
 *	\param step step size in X/Y direction in pixels
 *	\param border number of border pixels
 *
 *	Computes the same as buildDetCUDA() but uses shared memory to cut down
 *	global memory bandwidth usage. The function uses a brute force approach for
 *	sharing the data.
 *
 *	Execution configuration:
 *	  Thread block: { 16, 4, 6 } = 384
 *	  Block grid  : { number of pixels to process in X, same in Y }
 *
 *  intervals is assumed to be 4.
 *	The kernel has only been tested for step = 2 and will very likely only
 *	compute correct results with that value.
 */
__global__ void
buildDetCUDA_smem_bf(float *g_img, float *g_det,
	int i_width, int i_height, size_t i_widthStep,
	int intervals, int o, int step, int border)
{
	__shared__ float s_data[58*34];

	// Transform thread indices from { 16, 4, z } to { 64, z }
	unsigned int t_x = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int t_y = threadIdx.z;

	unsigned int base_idx_x = blockIdx.x * 32 + t_x;
	unsigned int base_idx_y = blockIdx.y * 8 + t_y;
	unsigned int base_idx   = base_idx_y * i_widthStep
							+ base_idx_x;
	unsigned int s_base_idx = t_y * 58 + t_x;
	unsigned int s_y_offset = blockDim.z * 58;
	unsigned int g_y_offset = blockDim.z * i_widthStep;
	if (t_x < 58
		&& base_idx_x < i_width)
	{
		// Load rows 0-5
		if (base_idx_y < i_height)
		{
			s_data[s_base_idx] = g_img[base_idx];
		}
		s_base_idx += s_y_offset;
		base_idx_y += blockDim.z;
		base_idx += g_y_offset;

		// Load rows 6-11
		if (base_idx_y < i_height)
		{
			s_data[s_base_idx] = g_img[base_idx];
		}
		s_base_idx += s_y_offset;
		base_idx_y += blockDim.z;
		base_idx += g_y_offset;

		// Load rows 12-17
		if (base_idx_y < i_height)
		{
			s_data[s_base_idx] = g_img[base_idx];
		}
		s_base_idx += s_y_offset;
		base_idx_y += blockDim.z;
		base_idx += g_y_offset;

		// Load rows 18-23
		if (base_idx_y < i_height)
		{
			s_data[s_base_idx] = g_img[base_idx];
		}
		s_base_idx += s_y_offset;
		base_idx_y += blockDim.z;
		base_idx += g_y_offset;

		// Load rows 24-29
		if (base_idx_y < i_height)
		{
			s_data[s_base_idx] = g_img[base_idx];
		}
		s_base_idx += s_y_offset;
		base_idx_y += blockDim.z;
		base_idx += g_y_offset;

		// Load rows 30-33
		if (base_idx_y < i_height
			&& t_y < 4)
		{
			s_data[s_base_idx] = g_img[base_idx];
		}
	}
	__syncthreads();

	// 384 threads are used for loading data into shared memory. For computing
	// determinants only 256 threads are needed.
	if (threadIdx.z >= 4) return;

	//Construct filter
	int i = threadIdx.z;
	const int l = dc_lobe_cache_unique[o * intervals + i];
	const int w = 3 * l;
	const int b = w / 2;

	int c = threadIdx.x * step + border;
	int r = threadIdx.y * step + border;

	int gbl_c = c + blockIdx.x * blockDim.x * step;
	int gbl_r = r + blockIdx.y * blockDim.y * step;
	if (gbl_c >= i_width - border || gbl_r >= i_height - border)
		return;

	// Caluclate box integrals
	float Dxx = BoxIntegral(s_data, 58, 34, 58, r - l + 1, c - b, 2 * l - 1, w)
			  - BoxIntegral(s_data, 58, 34, 58, r - l + 1, c - l / 2, 2 * l - 1, l) * 3.0f;
	float Dyy = BoxIntegral(s_data, 58, 34, 58, r - b, c - l + 1, w, 2 * l - 1)
			  - BoxIntegral(s_data, 58, 34, 58, r - l / 2, c - l + 1, l, 2 * l - 1) * 3.0f;
	float Dxy = BoxIntegral(s_data, 58, 34, 58, r - l, c + 1, l, l)
			  + BoxIntegral(s_data, 58, 34, 58, r + 1, c - l, l, l)
			  - BoxIntegral(s_data, 58, 34, 58, r - l, c - l, l, l)
			  - BoxIntegral(s_data, 58, 34, 58, r + 1, c + 1, l, l);

	// Normalise the filter responses with respect to their size
	float inverse_area = 1.0f / (w * w);
	Dxx *= inverse_area;
	Dyy *= inverse_area;
	Dxy *= inverse_area;

	// Get the sign of the laplacian
	const float lap_sign = (Dxx + Dyy >= 0.0f ? 1.0f : -1.0f);

	// Get the determinant of hessian response
	float determinant = (Dxx * Dyy - 0.81f * Dxy * Dxy);
	unsigned int cur_intvl = o * intervals + i;
	unsigned int save_idx = cur_intvl * i_width * i_height + (gbl_r * i_width + gbl_c);
	g_det[save_idx] = determinant < 0.0f ? 0.0f : lap_sign * determinant;
}

//-------------------------------------------------------

/////////////////////////////////////////////////////////
// Device functions
/////////////////////////////////////////////////////////

//! Computes the sum of pixels within the rectangle specified by the top-left start
//! co-ordinate (row, col) and size (rows, cols).
__device__ float
BoxIntegral(float *data, int width, int height, size_t widthStep,
	int row, int col, int rows, int cols)
{
	// The subtraction by one for row/col is because row/col is inclusive.
	int r1 = min(row, height) - 1;
	int c1 = min(col, width) - 1;
	int r2 = min(row + rows, height) - 1;
	int c2 = min(col + cols, width) - 1;

	float A, B, C, D;
	A = data[r1 * widthStep + c1];
	B = data[r1 * widthStep + c2];
	C = data[r2 * widthStep + c1];
	D = data[r2 * widthStep + c2];

	return max(0.f, A - B - C + D);
}

#endif /* CUDA_BUILDDETCUDA_CU */
