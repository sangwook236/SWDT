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

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

#include <cuda_runtime.h>
#include "cuda/helper_funcs.h"

struct cudaImage
{
    int    width;     /* Image width in pixels. */
    int    height;    /* Image height in pixels. */
    char*  data;      /* Pointer to aligned image data. */
    size_t widthStep; /* Size of aligned image row in bytes. */
};

inline void freeCudaImage(cudaImage *img)
{
	CUDA_SAFE_CALL( cudaFree(img->data) );
	delete img;
}

#endif /* CUDAIMAGE_H */
