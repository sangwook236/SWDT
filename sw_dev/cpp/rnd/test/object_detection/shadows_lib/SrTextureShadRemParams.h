// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef SRTEXTURESHADREMPARAMS_H_
#define SRTEXTURESHADREMPARAMS_H_

class SrTextureShadRemParams {

	public:
		double gainThreshold;
		int gaborKernelRadius;
		float gaborWavelength;
		float gaborAspectRatio;
		std::vector<float> gaborBandwidths;
		std::vector<float> gaborOrientations;
		std::vector<float> gaborPhases;
		int neighborhood;
		double distThreshold;
		bool cleanSrMask;

		SrTextureShadRemParams();
		virtual ~SrTextureShadRemParams();
};

#endif /* SRTEXTURESHADREMPARAMS_H_ */
