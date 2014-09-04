// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef SRTEXTURESHADREM_H_
#define SRTEXTURESHADREM_H_

#include <cxcore.h>
#include "SrTextureShadRemParams.h"
#include "Utils/GaborFilter.h"

class ConnCompGroup;

/**
 * Implemented from:
 *    Shadow detection for moving objects based on texture analysis
 *    Leone & Distante (PR 2007)
 *
 * The overcomplete Gabor dictionary is used for maximum accuracy
 */
class SrTextureShadRem {

	public:
		SrTextureShadRem(const SrTextureShadRemParams& params = SrTextureShadRemParams());
		virtual ~SrTextureShadRem();

		void removeShadows(const cv::Mat& frame, const cv::Mat& fg, const cv::Mat& bg, cv::Mat& srMask);

	private:
		SrTextureShadRemParams params;

		GaborFilter gaborFilter;
		std::vector<cv::Mat> frProjections;
		std::vector<cv::Mat> bgProjections;
		cv::Mat threshDistances;
		cv::Mat distanceMask;
		cv::Mat candidateShadows;
		cv::Mat distances;
		cv::Mat shadows;

		void extractCandidateShadowPixels(const cv::Mat& grayFrame, const ConnCompGroup& fg, const cv::Mat& grayBg,
				cv::Mat& candidateShadows);
		void getShadows(const cv::Mat& grayFrame, const cv::Mat& candidateShadows, const cv::Mat& grayBg,
				cv::Mat& shadows);
};

#endif /* SRTEXTURESHADREM_H_ */
