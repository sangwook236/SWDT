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

#ifndef CONVERTRGBA2GRAYFLOATGPU_H
#define CONVERTRGBA2GRAYFLOATGPU_H

/**	\brief Convert an RGBA image to gray
 *	\param d_dst device pointer to image data
 *	\param d_dst_pitch pitch used to access d_color_data (in bytes)
 *	\param d_src device pointer to save the red channel
 *	\param d_src_pitch pitch used to access d_red_data (in bytes)
 *	\param red_mask bit mask used to isolate the red color channel of a pixel
 *	\param green_mask bit mask used to isolate the green color channel of a pixel
 *	\param blue_mask bit mask used to isolate the blue color channel of a pixel
 *	\param red_shift number of bits between red color channel and bit 0
 *	\param green_shift number of bits between green color channel and bit 0
 *	\param blue_shift number of bits between blue color channel and bit 0
 *	\param img_width image width
 *	\param img_height image height
 *
 *	The function reads 32-bit pixels from d_src and converts them to
 *	gray-scale values using the given mask and shift parameters. The
 *	result is converted to float in the range [0.0, 1.0].
 */
void convertRGBA2GrayFloatGPU(
	float *d_dst, size_t d_dst_pitch,
	const unsigned int *d_src, size_t d_src_pitch,
	unsigned int red_mask, unsigned int green_mask, unsigned int blue_mask,
	unsigned char red_shift, unsigned char green_shift, unsigned char blue_shift,
	unsigned int img_width, unsigned int img_height);

#endif /* CONVERTRGBA2GRAYFLOATGPU_H */
