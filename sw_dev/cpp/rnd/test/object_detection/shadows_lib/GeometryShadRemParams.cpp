// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include "GeometryShadRemParams.h"

GeometryShadRemParams::GeometryShadRemParams() {
	smoothFactor = 4;
	headThreshRatio = 2;
	minHeadSeq = 4;
	maxEdgeDistance = 4;
	edgeThreshRatio = 2;
	minEdgeSeq = 3;
	bottomShiftRatio = 8;
	gWeight = 0.7;
	sRelativeWeight = 0.2;
	thresholdScale = 0.4;
	cleanSrMask = false;
}

GeometryShadRemParams::~GeometryShadRemParams() {
}
