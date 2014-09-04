// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef CHROMACITYSHADREM_H_
#define CHROMACITYSHADREM_H_

#include <cxcore.h>
#include "ChromacityShadRemParams.h"

class ConnCompGroup;

/**
 * Implemented from:
 *    Detecting moving objects, ghosts, and shadows in video streams
 *    Cucchiara et al. (PAMI 2003)
 *
 * Extended to use observation windows as proposed in:
 *    Cast shadow segmentation using invariant color features
 *    Salvador et al. (CVIU 2004)
 */
class ChromacityShadRem {

	public:
		ChromacityShadRem(const ChromacityShadRemParams& params = ChromacityShadRemParams());
		virtual ~ChromacityShadRem();

		void removeShadows(const cv::Mat& frame, const cv::Mat& fg, const cv::Mat& bg, cv::Mat& srMask);

	private:
		ChromacityShadRemParams params;

		void extractDarkPixels(const cv::Mat& hsvFrame, const ConnCompGroup& fg, const cv::Mat& hsvBg,
				ConnCompGroup& darkPixels);
		void extractShadows(const cv::Mat& hsvFrame, const ConnCompGroup& darkPixels, const cv::Mat& hsvBg,
				ConnCompGroup& shadows);
};

#endif /* CHROMACITYSHADREM_H_ */
