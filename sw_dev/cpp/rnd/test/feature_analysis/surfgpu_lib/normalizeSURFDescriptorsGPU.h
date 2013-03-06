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

#ifndef NORMALIZESURFDESCRIPTORSGPU_H
#define NORMALIZESURFDESCRIPTORSGPU_H

#include "surf_cudaipoint.h"

/**	\brief Normalize SURF descriptors
 *	\param d_ipoints device pointer to interest points
 *	\param num_ipoints number of interest points at d_ipoints
 */
void normalizeSURFDescriptorsGPU(surf_cudaIpoint *d_ipoints, size_t num_ipoints);

#endif /* NORMALIZESURFDESCRIPTORSGPU_H */
