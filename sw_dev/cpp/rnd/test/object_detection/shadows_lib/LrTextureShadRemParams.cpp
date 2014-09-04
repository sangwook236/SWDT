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
#include "LrTextureShadRemParams.h"

LrTextureShadRemParams::LrTextureShadRemParams() {
	avgSatThresh = 35;
	hThreshLowSat = 76;
	hThreshHighSat = 62;
	sThreshLowSat = 36;
	sThreshHighSat = 93;
	avgAttenThresh = 1.58;
	vThreshUpperLowAtten = 1;
	vThreshUpperHighAtten = 0.99;
	vThreshLowerLowAtten = 0.6;
	vThreshLowerHighAtten = 0.21;
	avgPerimThresh = 100;
	edgeDiffRadius = 1;
	borderDiffRadius = 0;
	splitIncrement = 1;
	splitRadius = 1;
	cannyThresh1 = 72;
	cannyThresh2 = 94;
	cannyApertureSize = 3;
	cannyL2Grad = true;

	minCorrPoints = 9;
	maxCorrRounds = 1;
	corrBorder = 1;
	gradScales = 1;
	gradMagThresh = 6;
	gradAttenThresh = 0.1;
	gradDistThresh = CV_PI / 10;
	gradCorrThreshLowAtten = 0.2;
	gradCorrThreshHighAtten = 0.1;

	cleanShadows = true;
	fillShadows = true;
	minShadowPerim = 35;
	cleanSrMask = true;
	fillSrMask = true;
}

LrTextureShadRemParams::~LrTextureShadRemParams() {
}
