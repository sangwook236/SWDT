// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef LRTEXTURESHADREMPARAMS_H_
#define LRTEXTURESHADREMPARAMS_H_

class LrTextureShadRemParams {

	public:
		float avgSatThresh;
		int hThreshLowSat;
		int hThreshHighSat;
		int sThreshLowSat;
		int sThreshHighSat;
		float avgAttenThresh;
		float vThreshUpperLowAtten;
		float vThreshUpperHighAtten;
		float vThreshLowerLowAtten;
		float vThreshLowerHighAtten;
		float avgPerimThresh;
		int edgeDiffRadius;
		int borderDiffRadius;
		int splitIncrement;
		int splitRadius;
		float cannyThresh1;
		float cannyThresh2;
		int cannyApertureSize;
		bool cannyL2Grad;

		int minCorrPoints;
		int maxCorrRounds;
		int corrBorder;
		int gradScales;
		float gradMagThresh;
		float gradAttenThresh;
		float gradDistThresh;
		float gradCorrThreshLowAtten;
		float gradCorrThreshHighAtten;

		bool cleanShadows;
		bool fillShadows;
		int minShadowPerim;
		bool cleanSrMask;
		bool fillSrMask;

		LrTextureShadRemParams();
		virtual ~LrTextureShadRemParams();
};

#endif /* LRTEXTURESHADREMPARAMS_H_ */
