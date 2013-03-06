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

#ifndef NONMAXGPU_H
#define NONMAXGPU_H

#include "fasthessian_cudaipoint.h"

/**	\brief Upload lobe_map to GPU
 *	\param h_lobe_map host pointer to lobe map array
 *	\param size number of bytes to copy from h_lobe_map
 *
 *	This function must be called once before calling nonMaxGPU().
 */
void prepare_nonMaxGPU(const int *h_lobe_map, size_t size);

/**	\brief Find interest points using the given determinants and perform NMS
 *	\param d_det device pointer to determinants
 *	\param det_width width of d_det in number of elements
 *	\param det_height height of d_det in number of elements
 *	\param det_width_step number of bytes in a row of d_det
 *	\param d_points device pointer to save interest points to
 *	\param border_cache host pointer to border sizes of the octaves
 *	\param octaves number of octaves to compute
 *	\param intervals number of intervals per octave to compute
 *	\param init_sample initial sampling step size in X/Y in pixels
 *	\param thres threshold for interest point detection
 *
 *	prepare_nonMaxGPU() must be called once before calling this function.
 */
void nonMaxGPU(float *d_det, unsigned int det_width, unsigned int det_height,
			   size_t det_width_step, fasthessian_cudaIpoint *d_points,
			   const int *border_cache, int octaves, int intervals,
			   int init_sample, float thres);

#endif /* NONMAXGPU_H */
