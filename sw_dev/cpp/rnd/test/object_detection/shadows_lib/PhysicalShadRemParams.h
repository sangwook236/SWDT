// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef PHYSICALSHADREMPARAMS_H_
#define PHYSICALSHADREMPARAMS_H_

#include "utils/GaussianMixtureModel.h"

class PhysicalShadRemParams {

	public:
		double coneAngle;
		double coneR1;
		double coneR2;
		double weightSmootTerm;
		bool learnBorders;
		int gmmGaussians;
		double gmmInitVar;
		double gmmMinVar;
		float gmmStdThreshold;
		bool gmmWinnerTakesAll;
		float gmmLearningRate;
		GaussianMixtureModel::SortMode gmmSortMode;
		bool gmmFitLogistic;
		double gmmAccumWeightThresh;
		double postThresh;
		bool cleanShadows;
		bool dilateShadows;
		bool cleanSrMask;

		PhysicalShadRemParams();
		virtual ~PhysicalShadRemParams();
};

#endif /* PHYSICALSHADREMPARAMS_H_ */
