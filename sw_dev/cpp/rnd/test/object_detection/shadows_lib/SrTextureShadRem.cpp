// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include <cv.h>
#include "SrTextureShadRem.h"
#include "utils/ConnCompGroup.h"

SrTextureShadRem::SrTextureShadRem(const SrTextureShadRemParams& params) {
	this->params = params;

	gaborFilter.createKernels(params.gaborKernelRadius, params.gaborWavelength, params.gaborAspectRatio,
			params.gaborBandwidths, params.gaborOrientations, params.gaborPhases);
}

SrTextureShadRem::~SrTextureShadRem() {
}

void SrTextureShadRem::removeShadows(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, cv::Mat& srMask) {
	ConnCompGroup fg(fgMask);
	fg.mask.copyTo(srMask);

	cv::Mat grayFrame, grayBg;
	cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);
	cv::cvtColor(bg, grayBg, CV_BGR2GRAY);

	extractCandidateShadowPixels(grayFrame, fg, grayBg, candidateShadows);
	getShadows(grayFrame, candidateShadows, grayBg, shadows);

	srMask.setTo(0, shadows);

	if (params.cleanSrMask) {
		ConnCompGroup ccg;
		ccg.update(srMask, true, true);
		ccg.mask.copyTo(srMask);
	}
}

void SrTextureShadRem::extractCandidateShadowPixels(const cv::Mat& grayFrame, const ConnCompGroup& fg,
		const cv::Mat& grayBg, cv::Mat& candidateShadows) {
	candidateShadows.create(grayFrame.size(), CV_8U);
	candidateShadows.setTo(cv::Scalar(0));

	for (int cc = 0; cc < (int) fg.comps.size(); ++cc) {
		const ConnComp& object = fg.comps[cc];

		for (int p = 0; p < (int) object.pixels.size(); ++p) {
			int x = object.pixels[p].x;
			int y = object.pixels[p].y;

			double frVal = grayFrame.at<uchar>(y, x);
			double bgVal = grayBg.at<uchar>(y, x);

			double gain = 0;
			if (frVal < bgVal) {
				gain = 1 - (frVal / bgVal) / (bgVal - frVal);
			}

			if (gain > params.gainThreshold) {
				candidateShadows.at<uchar>(y, x) = 255;
			}
		}
	}
}

void SrTextureShadRem::getShadows(const cv::Mat& grayFrame, const cv::Mat& candidateShadows, const cv::Mat& grayBg,
		cv::Mat& shadows) {
	gaborFilter.filter(grayFrame, frProjections, candidateShadows, params.neighborhood);
	gaborFilter.filter(grayBg, bgProjections, candidateShadows, params.neighborhood);

	GaborFilter::getDistance(frProjections, bgProjections, distances, candidateShadows);
	cv::threshold(distances, threshDistances, params.distThreshold, 255, cv::THRESH_BINARY_INV);
	threshDistances.convertTo(distanceMask, CV_8U);

	shadows.create(grayFrame.size(), CV_8U);
	shadows.setTo(cv::Scalar(0));
	distanceMask.copyTo(shadows, candidateShadows);
}
