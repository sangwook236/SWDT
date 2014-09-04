// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include <cxcore.h>
#include "SrTextureShadRemParams.h"

SrTextureShadRemParams::SrTextureShadRemParams() {
	gainThreshold = 0.5;
	gaborKernelRadius = 4;
	gaborWavelength = 16;
	gaborAspectRatio = 0.5;
	gaborBandwidths.push_back(0.4);
	gaborBandwidths.push_back(0.8);
	gaborBandwidths.push_back(1.6);
	gaborOrientations.push_back(0);
	gaborOrientations.push_back(1 * CV_PI / 8);
	gaborOrientations.push_back(2 * CV_PI / 8);
	gaborOrientations.push_back(3 * CV_PI / 8);
	gaborOrientations.push_back(4 * CV_PI / 8);
	gaborOrientations.push_back(5 * CV_PI / 8);
	gaborOrientations.push_back(6 * CV_PI / 8);
	gaborOrientations.push_back(7 * CV_PI / 8);
	gaborPhases.push_back(0);
	gaborPhases.push_back(CV_PI / 2);
	neighborhood = 1;
	distThreshold = 0.166;
	cleanSrMask = false;
}

SrTextureShadRemParams::~SrTextureShadRemParams() {
}
