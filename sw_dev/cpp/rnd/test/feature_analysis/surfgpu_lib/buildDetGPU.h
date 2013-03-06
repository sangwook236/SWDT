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

#ifndef BUILDDETGPU_H
#define BUILDDETGPU_H

#include "cudaimage.h"

/**	\brief Upload lobe_cache_unique array to GPU
 *	\param lobe_cache_unique host pointer to lobe_cache_unique
 *	\param size number bytes to copy
 *
 *	Needs to be called once before calling buildDetGPU().
 */
void prepare_buildDetGPU(const int *lobe_cache_unique, size_t size);

/**	\brief Compute determinants on the GPU
 *	\param d_img integral image
 *	\param d_det device pointer to save determinants to
 *	\param border_cache host pointer to border_cache
 *	\param octaves number of octave to compute
 *	\param intervals number of intervals to compute per octave
 *	\param init_sample initial sampling step size
 *
 *	prepare_buildDetGPU() must be called once before calling this function.
 */
void buildDetGPU(cudaImage *d_img, float *d_det, const int *border_cache,
				 int octaves, int intervals, int init_sample);

#endif /* BUILDDETGPU_H */
