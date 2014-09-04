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
#include "PhysicalShadRemParams.h"

PhysicalShadRemParams::PhysicalShadRemParams() {
	coneAngle = CV_PI / 20;
	coneR1 = 0.3;
	coneR2 = 1;
	weightSmootTerm = 4;
	learnBorders = false;
	gmmGaussians = 5;
	gmmInitVar = 30;
	gmmMinVar = 1;
	gmmStdThreshold = 2;
	gmmWinnerTakesAll = false;
	gmmLearningRate = 0.1;
	gmmSortMode = GaussianMixtureModel::SORT_BY_WEIGHT;
	gmmFitLogistic = false;
	gmmAccumWeightThresh = 0.7;
	postThresh = 0.15;
	cleanShadows = false;
	dilateShadows = false;
	cleanSrMask = false;
}

PhysicalShadRemParams::~PhysicalShadRemParams() {
}
