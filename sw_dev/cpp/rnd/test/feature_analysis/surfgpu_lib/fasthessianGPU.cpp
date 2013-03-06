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

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

//--S [] 2013/03/06: Sang-Wook Lee
//#include "integral.h"
//#include "ipoint.h"
#include "integralGPU.h"
#include "ipointGPU.h"
//--E [] 2013/03/06: Sang-Wook Lee

#include "cuda/helper_funcs.h"
#include "buildDetGPU.h"
#include "nonMaxGPU.h"

//--S [] 2013/03/06: Sang-Wook Lee
//#include "fasthessian.h"
#include "fasthessianGPU.h"
//--E [] 2013/03/06: Sang-Wook Lee
#include "defines.h"

using namespace std;

//--S [] 2013/03/06: Sang-Wook Lee
namespace surfgpu {
//--E [] 2013/03/06: Sang-Wook Lee

//-------------------------------------------------------
// pre calculated lobe sizes
static const int lobe_cache [] = {3,5,7,9,5,9,13,17,9,17,25,33,17,33,49,65};
static const int lobe_cache_unique [] = {3,5,7,9,13,17,25,33,49,65};
static const int lobe_map [] = {0,1,2,3,1,3,4,5,3,5,6,7,5,7,8,9};
static const int border_cache [] = {14,26,50,98};

//-------------------------------------------------------

//! Destructor
FastHessian::~FastHessian()
{
}

//-------------------------------------------------------

//! Constructor without image
FastHessian::FastHessian(std::vector<Ipoint> &ipts,
                         const int octaves, const int intervals, const int init_sample,
                         const float thres)
						 : d_img(NULL), i_width(0), i_height(0), d_points(NULL),
						   h_points(NULL), ipts(ipts), d_det(NULL), det_pitch(0)
{
  // Save parameter set
  saveParameters(octaves, intervals, init_sample, thres);

  initGPU();
}

//-------------------------------------------------------

//! Constructor with image
FastHessian::FastHessian(cudaImage *d_img,
						 std::vector<Ipoint> &ipts,
                         const int octaves, const int intervals, const int init_sample,
                         const float thres)
						 : d_img(NULL), i_width(0), i_height(0), d_points(NULL),
						   h_points(NULL), ipts(ipts), d_det(NULL), det_pitch(0)
{
  // Save parameter set
  saveParameters(octaves, intervals, init_sample, thres);

  // Set the current image
  setIntImage(d_img);

  initGPU();
}

//-------------------------------------------------------

//! Save the parameters
void FastHessian::saveParameters(const int octaves, const int intervals,
                                 const int init_sample, const float thres)
{
  // Initialise variables with bounds-checked values
  this->octaves =
    (octaves > 0 && octaves <= 4 ? octaves : OCTAVES);
  this->intervals =
    (intervals > 0 && intervals <= 4 ? intervals : INTERVALS);
  this->init_sample =
    (init_sample > 0 && init_sample <= 6 ? init_sample : INIT_SAMPLE);
  this->thres = (thres >= 0 ? thres : THRES);
}


//-------------------------------------------------------

//! Set or re-set the integral image source
void FastHessian::setIntImage(cudaImage *d_img)
{
  // Change the source image
  this->d_img = d_img;

  // Redefine width, height and det map only if image has changed size
  if (d_img->width != this->i_width || d_img->height != this->i_height)
  {
    this->i_width = d_img->width;
    this->i_height = d_img->height;
  }
}

//-------------------------------------------------------

//! Find the image features and write into vector of features
void FastHessian::getIpoints()
{
	// Clear the vector of exisiting ipts
	ipts.clear();

	// Determine if the requested number of octaves can actually be computed on
	// the given image and adjust if necessary.
	adjd_octaves = 0;
	for (unsigned int i = 0; i < OCTAVES; i++)
	{
		int border_pixels = 2 * border_cache[i];
		adjd_octaves += (i_width > border_pixels) && (i_height > border_pixels);
	}
	adjd_octaves = std::min(adjd_octaves, octaves);

	// If the given image is too small to compute any octaves on it we bail out.
	if (adjd_octaves == 0) return;

	// Calculate approximated determinant of hessian values
	buildDet();

	// Run 3x3x3 Non-maximum suppression on device
	nonMaxGPU(d_det, d_img->width, d_img->height, d_img->width,
		d_points, border_cache, adjd_octaves, intervals,
		init_sample, thres);

	// Download interest points to host
	size_t points_size = ((i_width * i_height) / IMG_SIZE_DIVISOR)
						* sizeof(fasthessian_cudaIpoint);
	CUDA_SAFE_CALL( cudaMemcpy(h_points, d_points, points_size,
							  cudaMemcpyDeviceToHost) );

	collectPoints(h_points);

	// Free device memory
	CUDA_SAFE_CALL( cudaFree(d_det) );
	CUDA_SAFE_CALL( cudaFree(d_points) );

	delete[] h_points;
}

//-------------------------------------------------------

//! Calculate determinant of hessian responses
void FastHessian::buildDet()
{
	size_t max_num_ipoints = (i_width * i_height) / IMG_SIZE_DIVISOR;
	size_t points_size = max_num_ipoints * sizeof(fasthessian_cudaIpoint);
	h_points = new fasthessian_cudaIpoint[max_num_ipoints];
	std::memset(h_points, 0, points_size); // XXX: needed?

	// Calculate sizes
	unsigned int total_num_intervals = intervals + (adjd_octaves - 1) * intervals / 2;
	size_t det_size = total_num_intervals * i_width * i_height * sizeof(float);

	CUDA_SAFE_CALL( cudaMalloc((void**)&d_det, det_size) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_points, points_size) );
	CUDA_SAFE_CALL( cudaMemset(d_det, 0, det_size) );
	CUDA_SAFE_CALL( cudaMemset(d_points, 0, points_size) );

	// Calculate the determinants on the GPU
	buildDetGPU(d_img, d_det, border_cache, adjd_octaves, intervals, init_sample);
}

//-------------------------------------------------------

void FastHessian::collectPoints(fasthessian_cudaIpoint *points) {
	for (unsigned int i = 0; i < (i_width * i_height) / IMG_SIZE_DIVISOR; i++)
	{
		const fasthessian_cudaIpoint &p = points[i];

		if (p.x != 0.0f || p.y != 0.0f || p.scale != 0.0f || p.laplacian != 0)
		{
			Ipoint ipt;
			ipt.x = p.x;
			ipt.y = p.y;
			ipt.scale = p.scale;
			ipt.laplacian = p.laplacian;
			ipts.push_back(ipt);
		}
	}
}

//-------------------------------------------------------

void FastHessian::initGPU()
{
	prepare_buildDetGPU(lobe_cache_unique, sizeof(lobe_cache_unique));
	prepare_nonMaxGPU(lobe_map, sizeof(lobe_map));
}

//--S [] 2013/03/06: Sang-Wook Lee
}  // namespace surfgpu
//--E [] 2013/03/06: Sang-Wook Lee
