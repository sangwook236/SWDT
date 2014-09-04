// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include "ChromacityShadRemParams.h"

ChromacityShadRemParams::ChromacityShadRemParams() {
	winSize = 1;
	cleanSrMask = false;
	hThresh = 48;
	sThresh = 40;
	vThreshUpper = 1;
	vThreshLower = 0.3;
}

ChromacityShadRemParams::~ChromacityShadRemParams() {
}
