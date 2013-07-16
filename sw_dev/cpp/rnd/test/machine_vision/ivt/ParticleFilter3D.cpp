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
// Filename:  ParticleFilter3D.cpp
// Author:    Pedram Azad
// Date:      2005
// ****************************************************************************


// ****************************************************************************
// Includes
// ****************************************************************************

#include "ParticleFilter3D.h"
#include "Helpers/helpers.h"
#include "Image/ImageProcessor.h"
#include "Image/ByteImage.h"
#include "Image/IntImage.h"

#include <math.h>
#include <float.h>


// ****************************************************************************
// Defines
// ****************************************************************************

#define VELOCITY_FACTOR			0.7



// ****************************************************************************
// Constructor / Destructor
// ****************************************************************************

CParticleFilter3D::CParticleFilter3D(int nParticles, int nImageWidth, int nImageHeight, int k) : CParticleFilterFramework(nParticles, DIMENSION_3D)
{
	m_nParticles = nParticles;

	width = nImageWidth;
	height = nImageHeight;
	
	m_pSegmentedImage = 0;
	m_pSummedAreaTable = new CIntImage(width, height);
	
	m_ppProbabilities[0] = new float[m_nParticles];
	m_ppProbabilities[1] = new float[m_nParticles];
	m_nParticleIndex = 0;
	
	InitParticles(width / 2, height / 2, k);
}

CParticleFilter3D::~CParticleFilter3D()
{
	delete m_pSummedAreaTable;
	delete [] m_ppProbabilities[0];
	delete [] m_ppProbabilities[1];
}


// ****************************************************************************
// Methods
// ****************************************************************************

void CParticleFilter3D::SetImage(const CByteImage *pSegmentedImage)
{
	m_pSegmentedImage = pSegmentedImage;
	ImageProcessor::CalculateSummedAreaTable(pSegmentedImage, m_pSummedAreaTable);
}

void CParticleFilter3D::InitParticles(int x, int y, int k)
{
	int i;

	// init particle related attributes
	for (i = 0; i < m_nParticles; i++)
	{
		// particle positions
		s[i][0] = x;
		s[i][1] = y;
		s[i][2] = k;

		// probability for each particle
		pi[i] = 1.0 / m_nParticles;
	}

	// initialize configurations
	for (i = 0; i < DIMENSION_3D; i++)
	{
		mean_configuration[i] = s[0][i];
		last_configuration[i] = s[0][i];
	}

	c_total = 1.0;

	// maximum offset for next configuration
	sigma[0] = 10;
	sigma[1] = 10;
	sigma[2] = 2;
}

double CParticleFilter3D::CalculateProbability(bool bSeparateCall)
{
	const int k2 = 2 * model.k + 1;
	
	// optimized method using a summed area table (also called integral image)
	const int sum = ImageProcessor::GetAreaSum(m_pSummedAreaTable, model.x - model.k, model.y - model.k, model.x + model.k, model.y + model.k);
	
	if (bSeparateCall)
		return expf(-20.0f * (1.0f - sum / float(k2 * k2 * 255)));
		
	m_ppProbabilities[0][m_nParticleIndex] = -sum / float(k2 * k2 * 255);
	m_ppProbabilities[1][m_nParticleIndex] = -sum / float(255);
	m_nParticleIndex++;

	return 0;
}

void CParticleFilter3D::UpdateModel(int nParticle)
{
	model.x = int(s[nParticle][0] + 0.5);
	model.y = int(s[nParticle][1] + 0.5);
	model.k = int(s[nParticle][2] + 0.5);
}

void CParticleFilter3D::PredictNewBases(double dSigmaFactor)
{
	int nNewIndex = 0;
	
	for (nNewIndex = 0; nNewIndex < m_nParticles; nNewIndex++)
	{
		int nOldIndex = PickBaseSample();

		const double xx = s[nOldIndex][0] + VELOCITY_FACTOR * (mean_configuration[0] - last_configuration[0]) + dSigmaFactor * sigma[0] * gaussian_random();
		const double yy  = s[nOldIndex][1] + VELOCITY_FACTOR * (mean_configuration[1] - last_configuration[1]) + dSigmaFactor * sigma[1] * gaussian_random();
		const double kk  = s[nOldIndex][2] + VELOCITY_FACTOR * (mean_configuration[2] - last_configuration[2]) + dSigmaFactor * sigma[2] * gaussian_random();
		
		const int x = int(xx + 0.5);
		const int y = int(yy + 0.5);
		const int k = int(kk + 0.5);

		if (k < 5 || k > 100 || x - k < 0 || x + k >= width || y - k < 0 || y + k >= height)
		{
			s_temp[nNewIndex][0] = s_temp[nOldIndex][0];
			s_temp[nNewIndex][1] = s_temp[nOldIndex][1];
			s_temp[nNewIndex][2] = s_temp[nOldIndex][2];
		}
		else
		{
			s_temp[nNewIndex][0] = xx;
			s_temp[nNewIndex][1] = yy;
			s_temp[nNewIndex][2] = kk;
		}

	}

	// switch old/new
	double **temp = s_temp;
	s_temp = s;
	s = temp;
}

void CParticleFilter3D::CalculateFinalProbabilities()
{
	m_nParticleIndex = 0;
	
	float min = FLT_MAX;
	float max = FLT_MIN;
	int i;
	
	for (i = 0; i < 2; i++)
	{
		int j;
		
		for (j = 0; j < m_nParticles; j++)
		{
			if (m_ppProbabilities[i][j] < min)
				min = m_ppProbabilities[i][j];
					
			if (m_ppProbabilities[i][j] > max)
				max = m_ppProbabilities[i][j];
		}

		if (max != min)
		{
			const float fFactor = 1.0f / (max - min);
			
			for (j = 0; j < m_nParticles; j++)
				m_ppProbabilities[i][j] = (m_ppProbabilities[i][j] - min) * fFactor;
		}
		else
		{
			for (j = 0; j < m_nParticles; j++)
				m_ppProbabilities[i][j] = 1.0f / m_nParticles;
		}
	}
	
	for (i = 0; i < m_nParticles; i++)
		pi[i] = expf(-20.0f * (5 * m_ppProbabilities[0][i] + m_ppProbabilities[1][i]));
}
