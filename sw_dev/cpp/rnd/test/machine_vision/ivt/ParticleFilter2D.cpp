// ****************************************************************************
// This file is part of the Integrating Vision Toolkit (IVT).
//
// The IVT is maintained by the Karlsruhe Institute of Technology (KIT)
// (www.kit.edu) in cooperation with the company Keyetech (www.keyetech.de).
//
// Copyright (C) 2013 Karlsruhe Institute of Technology (KIT).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the KIT nor the names of its contributors may be
//    used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE KIT AND CONTRIBUTORS “AS IS” AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE KIT OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ****************************************************************************
// ****************************************************************************
// Filename:  ParticleFilter2D.cpp
// Author:    Pedram Azad
// Date:      2005
// ****************************************************************************


// ****************************************************************************
// Includes
// ****************************************************************************

#include "ParticleFilter2D.h"
#include "Helpers/helpers.h"
#include "Image/ImageProcessor.h"
#include "Image/ByteImage.h"
#include "Image/IntImage.h"
#include <math.h>


// ****************************************************************************
// Defines
// ****************************************************************************

#define VELOCITY_FACTOR			0.7



// ****************************************************************************
// Constructor / Destructor
// ****************************************************************************

CParticleFilter2D::CParticleFilter2D(int nParticles, int nImageWidth, int nImageHeight, int k) : CParticleFilterFramework(nParticles, DIMENSION_2D)
{
	m_nParticles = nParticles;

	width = nImageWidth;
	height = nImageHeight;
	
	m_pSegmentedImage = 0;
	m_pSummedAreaTable = new CIntImage(width, height);

	model.k = k;

	InitParticles(width / 2, height / 2);
}

CParticleFilter2D::~CParticleFilter2D()
{
	delete m_pSummedAreaTable;
}


// ****************************************************************************
// Methods
// ****************************************************************************

void CParticleFilter2D::SetImage(const CByteImage *pSegmentedImage)
{
	m_pSegmentedImage = pSegmentedImage;
	ImageProcessor::CalculateSummedAreaTable(pSegmentedImage, m_pSummedAreaTable);
}

void CParticleFilter2D::InitParticles(int x, int y)
{
	int i;

	// init particle related attributes
	for (i = 0; i < m_nParticles; i++)
	{
		// particle positions
		s[i][0] = x;
		s[i][1] = y;

		// probability for each particle
		pi[i] = 1.0 / m_nParticles;
	}

	// initialize configurations
	for (i = 0; i < DIMENSION_2D; i++)
	{
		mean_configuration[i] = s[0][i];
		last_configuration[i] = s[0][i];
	}

	c_total = 1.0;

	// limits for positions
	lower_limit[0] = model.k;
	lower_limit[1] = model.k;
	upper_limit[0] = width - model.k - 1;
	upper_limit[1] = height - model.k - 1;

	// maximum offset for next configuration
	sigma[0] = 10;
	sigma[1] = 10;
}

double CParticleFilter2D::CalculateProbability(bool bSeparateCall)
{
	const int k2 = 2 * model.k + 1;
	
	#if 0
	const int diff = width - k2;
	unsigned char *pixels = m_pSegmentedImage->pixels;
	int offset = (model.y - model.k) * width + (model.x - model.k);
	int sum = 0;

	for (int y = 0; y < k2; y++, offset += diff)
		for (int x = 0; x < k2; x++, offset++)
			sum += pixels[offset];*/
	#else
	// equivalent optimized method using a summed area table (also called integral image)
	const int sum = ImageProcessor::GetAreaSum(m_pSummedAreaTable, model.x - model.k, model.y - model.k, model.x + model.k, model.y + model.k);
	#endif

	return expf(-20.0f * (1.0f - sum / float(k2 * k2 * 255)));
}

void CParticleFilter2D::UpdateModel(int nParticle)
{
	model.x = int(s[nParticle][0] + 0.5);
	model.y = int(s[nParticle][1] + 0.5);
}

void CParticleFilter2D::PredictNewBases(double dSigmaFactor)
{
	int nNewIndex = 0;
	
	for (nNewIndex = 0; nNewIndex < m_nParticles; nNewIndex++)
	{
		int nOldIndex = PickBaseSample();

		const double xx = s[nOldIndex][0] + VELOCITY_FACTOR * (mean_configuration[0] - last_configuration[0]) + dSigmaFactor * sigma[0] * gaussian_random();
		const double yy  = s[nOldIndex][1] + VELOCITY_FACTOR * (mean_configuration[1] - last_configuration[1]) + dSigmaFactor * sigma[1] * gaussian_random();
		
		const int x = int(xx + 0.5);
		const int y = int(yy + 0.5);

		if (x < lower_limit[0] || x > upper_limit[0])
			s_temp[nNewIndex][0] = s_temp[nOldIndex][0];
		else
			s_temp[nNewIndex][0] = xx;

		if (y < lower_limit[1] || y > upper_limit[1])
			s_temp[nNewIndex][1] = s_temp[nOldIndex][1];
		else
			s_temp[nNewIndex][1] = yy;
	}

	// switch old/new
	double **temp = s_temp;
	s_temp = s;
	s = temp;
}
