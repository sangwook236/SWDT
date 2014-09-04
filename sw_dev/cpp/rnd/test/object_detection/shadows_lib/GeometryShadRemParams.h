// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef GEOMETRYSHADREMPARAMS_H_
#define GEOMETRYSHADREMPARAMS_H_

class GeometryShadRemParams {

	public:
		int smoothFactor;
		int headThreshRatio;
		int minHeadSeq;
		int maxEdgeDistance;
		int edgeThreshRatio;
		int minEdgeSeq;
		int bottomShiftRatio;
		double gWeight;
		double sRelativeWeight;
		double thresholdScale;
		bool cleanSrMask;

		GeometryShadRemParams();
		virtual ~GeometryShadRemParams();
};

#endif /* GEOMETRYSHADREMPARAMS_H_ */
