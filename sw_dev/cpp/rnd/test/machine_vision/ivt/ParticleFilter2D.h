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
// Filename:  ParticleFilter2D.h
// Author:    Pedram Azad
// Date:      2005
// ****************************************************************************


#ifndef _PARTICLE_FILTER_2D_H_
#define _PARTICLE_FILTER_2D_H_


// ****************************************************************************
// Necessary includes
// ****************************************************************************

#include "ParticleFilter/ParticleFilterFramework.h"


// ****************************************************************************
// Defines
// ****************************************************************************

#define DIMENSION_2D				2


// ****************************************************************************
// Forward declarations
// ****************************************************************************

class CByteImage;
class CIntImage;



// ****************************************************************************
// CParticleFilter2D
// ****************************************************************************

class CParticleFilter2D : public CParticleFilterFramework 
{	
public:
	struct Square2D
	{
		int x, y;
		int k;
	};


	// constructor
	CParticleFilter2D(int nParticles, int nImageWidth, int nImageHeight, int k);
	
	// destructor
	~CParticleFilter2D();
	

	// public methods
	void InitParticles(int x, int y);
	void SetImage(const CByteImage *pSegmentedImage);
	double CalculateProbability(bool bSeparateCall = true);


private:
	// private virtual methods from base class CParticleFilterFramework
	void UpdateModel(int nParticle);
	void PredictNewBases(double dSigmaFactor);
	

	// private attributes
	Square2D model;
	int m_nParticles;

	// image
	const CByteImage *m_pSegmentedImage;
	CIntImage *m_pSummedAreaTable;
	int width, height;
};



#endif /* _PARTICLE_FILTER_3D_H_ */
