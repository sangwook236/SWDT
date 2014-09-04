// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef CHROMACITYSHADREMPARAMS_H_
#define CHROMACITYSHADREMPARAMS_H_

class ChromacityShadRemParams {

	public:
		int winSize;
		bool cleanSrMask;
		int hThresh;
		int sThresh;
		float vThreshUpper;
		float vThreshLower;

		ChromacityShadRemParams();
		virtual ~ChromacityShadRemParams();
};

#endif /* CHROMACITYSHADREMPARAMS_H_ */
