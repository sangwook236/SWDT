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

#include <cudpp.h>

#include "strerror_cudpp.h"

const char*
strerror_cudpp(CUDPPResult err)
{
	switch (err)
	{
		case CUDPP_SUCCESS:
			return "CUDPP_SUCCESS";
		case CUDPP_ERROR_INVALID_HANDLE:
			return "CUDPP_ERROR_INVALID_HANDLE";
		case CUDPP_ERROR_ILLEGAL_CONFIGURATION:
			return "CUDPP_ERROR_ILLEGAL_CONFIGURATION";
		case CUDPP_ERROR_INVALID_PLAN:
			return "CUDPP_ERROR_INVALID_PLAN";
		case CUDPP_ERROR_INSUFFICIENT_RESOURCES:
			return "CUDPP_ERROR_INSUFFICIENT_RESOURCES";
		case CUDPP_ERROR_UNKNOWN:
		default:
			return "CUDPP_ERROR_UNKNOWN";
	}
}

