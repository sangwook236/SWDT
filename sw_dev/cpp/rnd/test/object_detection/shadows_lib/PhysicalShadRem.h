// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef PHYSICALSHADREM_H_
#define PHYSICALSHADREM_H_

#include <cxcore.h>
#include "PhysicalShadRemParams.h"
#include "Utils/ConnCompGroup.h"

/**
 * Implemented from:
 *    Moving cast shadow detection using physics-based features
 *    Huang & Chen (CVPR 2009)
 *
 * Note that this method requires a training phase where the removeShadows function is
 * called for a sequence of training frames
 */
class PhysicalShadRem {

	public:
		PhysicalShadRem(const PhysicalShadRemParams& params = PhysicalShadRemParams());
		virtual ~PhysicalShadRem();

		void removeShadows(const cv::Mat& frame, const cv::Mat& fg, const cv::Mat& bg, cv::Mat& srMask);

	private:
		static const float BGR2GRAY[];

		PhysicalShadRemParams params;

		GaussianMixtureModel gmm;
		cv::Mat features;
		ConnCompGroup candidateShadows;
		cv::Mat weights;
		cv::Mat gaussians;
		cv::Mat posteriors;
		cv::Mat shadows;

		static double angularDistance(const cv::Scalar vec1, const cv::Scalar vec2);
		void extractCandidateShadowPixels(const cv::Mat& frame, const ConnCompGroup& fg, const cv::Mat& bg,
				ConnCompGroup& candidateShadows);
		void getFeatures(const cv::Mat& frame, const ConnCompGroup& candidates, const cv::Mat& bg, cv::Mat& features);
		void getGradientWeights(const cv::Mat& frame, const ConnCompGroup& candidates, const cv::Mat& bg,
				cv::Mat& weights);
		void getShadows(const cv::Mat& features, const cv::Mat& weights, const ConnCompGroup& candidates,
				cv::Mat& shadows);
};

#endif /* PHYSICALSHADREM_H_ */
