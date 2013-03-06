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

#ifndef CUDA_HELPER_FUNCS_H
#define CUDA_HELPER_FUNCS_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// All of these were taken from the CUDA SDK and in some cases modified.

/**	\brief Divide 'a' by 'b' and round up
 *	\param a dividend
 *	\param b divisor
 */
int iDivUp(int a, int b);

/**	\brief Divide 'a' by 'b' and round down
 *	\param a dividend
 *	\param b divisor
 */
int iDivDown(int a, int b);

/**	\brief Align 'a' up to a multiple of 'b'
 *	\param a value to align
 *	\param b alignment
 */
int iAlignUp(int a, int b);

/**	\brief Align 'a' down to a multiple of 'b'
 *	\param a value to align
 *	\param b alignment
 */
int iAlignDown(int a, int b);


#if defined(_DEBUG)
#	define CUDA_SAFE_CALL(x) __cudaSafeCall(x, __FILE__, __LINE__)
#	define CUDA_CHECK_MSG(x) __cudaCheckMsg(x, __FILE__, __LINE__)
#else
#	define CUDA_SAFE_CALL(x) (x)
#	define CUDA_CHECK_MSG(x)
#endif

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
	if (err != cudaSuccess)
	{
		std::fprintf(stderr, "ERROR: %s:%i: __cudaSafeCall() %i: %s\n",
				file, line, err, cudaGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

inline void __cudaCheckMsg(const char *msg, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::fprintf(stderr, "ERROR: %s:%i: %s: %s (%d)",
				file, line, msg, cudaGetErrorString(err), err);
		std::exit(EXIT_FAILURE);
	}
}

#endif /* CUDA_HELPER_FUNCS_H */
