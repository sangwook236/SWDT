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

#ifndef TRANSPOSEGPU_H
#define TRANSPOSEGPU_H

/**	\brief Transpose a matrix of float values
 *	\param d_dst device pointer to destination matrix
 *	\param dst_pitch number of bytes in a row of d_dst
 *	\param d_src device pointer to source matrix
 *	\param src_pitch number of bytes in a row of d_src
 *	\param width matrix width
 *	\param height matrix height
 */
void transposeGPU(float *d_dst, size_t dst_pitch,
	float *d_src, size_t src_pitch,
	unsigned int width, unsigned int height);

#endif /* TRANSPOSEGPU_H */
