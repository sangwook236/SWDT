/*
 * Copyright (C) 2011      Andre Schulz, Florian Jung, Sebastian Hartte,
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

#ifndef CUDA_CUDPP_HELPER_FUNCS_H
#define CUDA_CUDPP_HELPER_FUNCS_H

#include <cstdlib>
#include <cstdio>

#include <cudpp.h>

#include "strerror_cudpp.h"

#ifdef _DEBUG
#	define CUDPP_SAFE_CALL(err)	__cudppSafeCall(err, __FILE__, __LINE__)
#else
#	define CUDPP_SAFE_CALL(err) (err)
#endif

inline void
__cudppSafeCall( CUDPPResult err, const char *file, const int line )
{
	if (CUDPP_SUCCESS != err)
	{
		std::fprintf(stderr, "ERROR: %s:%i: __cudppSafeCall() %d: %s\n",
				file, line, (int)err, strerror_cudpp(err));
		std::exit(EXIT_FAILURE);
	}
}

#endif /* CUDA_CUDPP_HELPER_FUNCS_H */
