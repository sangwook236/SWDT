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

#ifndef BUILDSURFDESCRIPTORSGPU_H
#define BUILDSURFDESCRIPTORSGPU_H

#include "surf_cudaipoint.h"

/**	\brief Upload gaussian look-up table to GPU
 *
 *	Needs to be called once before calling buildSURFDescriptorsGPU().
 */
void prepare_buildSURFDescriptorsGPU();

/**	\brief Bind texture reference to integral image CUDA array
 *	\param ca_intimg host pointer to the desired CUDA array
 *
 *	This function must be called everytime before calling
 *	buildSURFDescriptorsGPU().
 */
void prepare2_buildSURFDescriptorsGPU(cudaArray *ca_intimg);

/**	\brief Build SURF descriptors for interest points
 *	\param d_ipoints device pointer to interest points
 *	\param upright if != 0 compute upright version, otherwise standard version
 *	\param num_ipoints number of interest points at d_ipoints
 *
 *	prepare_buildSURFDescriptorsGPU() must be called once before calling this
 *	function.
 *	prepare2_buildSURFDescriptorsGPU() must always be called before calling
 *	this function.
 */
void buildSURFDescriptorsGPU(
	surf_cudaIpoint *d_ipoints, int upright, size_t num_ipoints);

#endif /* BUILDSURFDESCRIPTORSGPU_H */
