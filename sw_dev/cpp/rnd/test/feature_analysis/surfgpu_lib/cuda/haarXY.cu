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

#ifndef CUDA_HAARXY_CU
#define CUDA_HAARXY_CU

/*
 * This inline function is used by both the orientation and
 * description kernel to take samples from the source image.
 *
 * It computes the Haar X & Y response simultaneously.
 */
__device__ __inline__ void
haarXY(int sampleX, int sampleY, int roundedScale,
	   float *xResponse, float *yResponse, float gauss)
{
	float leftTop, middleTop, rightTop,
		  leftMiddle, rightMiddle,
		  leftBottom, middleBottom, rightBottom;

	int xmiddle = sampleX;
	int ymiddle = sampleY;
	int left = xmiddle - roundedScale;
	int right = xmiddle + roundedScale;
	int top = ymiddle - roundedScale;
	int bottom = ymiddle + roundedScale;

	leftTop = tex2D(integralImage,  left, top);
	leftMiddle = tex2D(integralImage,  left, ymiddle);
	leftBottom = tex2D(integralImage,  left, bottom);
	rightTop = tex2D(integralImage,  right, top);
	rightMiddle = tex2D(integralImage,  right, ymiddle);
	rightBottom = tex2D(integralImage,  right, bottom);
	middleTop = tex2D(integralImage,  xmiddle, top);
	middleBottom = tex2D(integralImage,  xmiddle, bottom);

	float upperHalf = leftTop - rightTop - leftMiddle + rightMiddle;
	float lowerHalf = leftMiddle - rightMiddle - leftBottom + rightBottom;
	*yResponse = gauss * (lowerHalf - upperHalf);

	float rightHalf = middleTop - rightTop - middleBottom + rightBottom;
	float leftHalf = leftTop - middleTop - leftBottom + middleBottom;
	*xResponse = gauss * (rightHalf - leftHalf);
}

#endif /* CUDA_HAARXY_CU */
