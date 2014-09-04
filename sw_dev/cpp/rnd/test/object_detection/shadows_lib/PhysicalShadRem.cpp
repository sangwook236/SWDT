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
#include "PhysicalShadRem.h"
#include "utils/ConnCompGroup.h"

const float PhysicalShadRem::BGR2GRAY[] = { 0.114f, 0.299f, 0.587f };

PhysicalShadRem::PhysicalShadRem(const PhysicalShadRemParams& params) {
	this->params = params;
}

PhysicalShadRem::~PhysicalShadRem() {
}

void PhysicalShadRem::removeShadows(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, cv::Mat& srMask) {
	ConnCompGroup fg(fgMask);
	fg.mask.copyTo(srMask);

	extractCandidateShadowPixels(frame, fg, bg, candidateShadows);
	getFeatures(frame, candidateShadows, bg, features);
	getGradientWeights(frame, candidateShadows, bg, weights);
	getShadows(features, weights, candidateShadows, shadows);

	if (params.cleanShadows) {
		ConnCompGroup ccg;
		ccg.update(shadows, true, true, 35);
		ccg.mask.copyTo(shadows);
	}

	if (params.dilateShadows) {
		cv::dilate(shadows, shadows, cv::Mat());
	}

	srMask.setTo(0, shadows);

	if (params.cleanSrMask) {
		ConnCompGroup ccg;
		ccg.update(srMask, true, true);
		ccg.mask.copyTo(srMask);
	}
}

double PhysicalShadRem::angularDistance(const cv::Scalar vec1, const cv::Scalar vec2) {
	double dotProduct = 0;
	double vec1SqrSum = 0;
	double vec2SqrSum = 0;
	for (int c = 0; c < 3; ++c) {
		dotProduct += vec1.val[c] * vec2.val[c];
		vec1SqrSum += vec1.val[c] * vec1.val[c];
		vec2SqrSum += vec2.val[c] * vec2.val[c];
	}

	return std::acos(dotProduct / std::sqrt(vec1SqrSum * vec2SqrSum));
}

void PhysicalShadRem::extractCandidateShadowPixels(const cv::Mat& frame, const ConnCompGroup& fg, const cv::Mat& bg,
		ConnCompGroup& candidateShadows) {
	cv::Mat mask(frame.size(), CV_8U, cv::Scalar(0));

	for (int cc = 0; cc < (int) fg.comps.size(); ++cc) {
		const ConnComp& object = fg.comps[cc];

		for (int p = 0; p < (int) object.pixels.size(); ++p) {
			int x = object.pixels[p].x;
			int y = object.pixels[p].y;

			const uchar* framePtr = frame.ptr(y) + x * 3;
			const uchar* bgPtr = bg.ptr(y) + x * 3;
			cv::Scalar vec1(framePtr[0], framePtr[1], framePtr[2]);
			cv::Scalar vec2(bgPtr[0], bgPtr[1], bgPtr[2]);

			double angle = angularDistance(vec1, vec2);
			double r = (cv::norm(vec1) * std::cos(angle)) / cv::norm(vec2);

			if (angle <= params.coneAngle && r >= params.coneR1 && r <= params.coneR2) {
				uchar* maskPtr = mask.ptr(y);
				maskPtr[x] = 255;
			}
		}
	}

	candidateShadows.update(mask);
}

void PhysicalShadRem::getFeatures(const cv::Mat& frame, const ConnCompGroup& candidates, const cv::Mat& bg,
		cv::Mat& features) {
	features.create(frame.size(), CV_8UC3);

	for (int cc = 0; cc < (int) candidates.comps.size(); ++cc) {
		const ConnComp& object = candidates.comps[cc];

		for (int p = 0; p < (int) object.pixels.size(); ++p) {
			int x = object.pixels[p].x;
			int y = object.pixels[p].y;

			const uchar* frPtr = frame.ptr(y) + x * 3;
			const uchar* bgPtr = bg.ptr(y) + x * 3;
			uchar* featuresPtr = features.ptr(y) + x * 3;

			cv::Scalar frToBgVec(bgPtr[0] - frPtr[0], bgPtr[1] - frPtr[1], bgPtr[2] - frPtr[2]);
			cv::Scalar bgVec(bgPtr[0], bgPtr[1], bgPtr[2]);
			double frToBgNorm = cv::norm(frToBgVec);
			double bgNorm = cv::norm(bgVec);

			double attenuation = std::min(frToBgNorm / bgNorm, 1.0);
			attenuation *= 255;
			featuresPtr[0] = attenuation;

			double grDirection = std::atan((double) frToBgVec[1] / frToBgVec[2]);
			grDirection = ((grDirection + CV_PI / 2) / CV_PI) * 255;
			featuresPtr[1] = grDirection;

			double bDirection = std::acos((double) frToBgVec[0] / frToBgNorm);
			bDirection = ((bDirection + CV_PI / 2) / CV_PI) * 255;
			featuresPtr[2] = bDirection;
		}
	}
}

void PhysicalShadRem::getGradientWeights(const cv::Mat& frame, const ConnCompGroup& candidates, const cv::Mat& bg,
		cv::Mat& weights) {
	weights.create(frame.size(), CV_32F);
	weights.setTo(cv::Scalar(0));

	for (int cc = 0; cc < (int) candidates.comps.size(); ++cc) {
		const ConnComp& object = candidates.comps[cc];

		for (int p = 0; p < (int) object.pixels.size(); ++p) {
			int x = object.pixels[p].x;
			int y = object.pixels[p].y;

			int dx = (x < (frame.cols - 1) ? 1 : -1);
			int dy = (y < (frame.rows - 1) ? 1 : -1);

			const uchar* frPtr = frame.ptr(y) + x * 3;
			const uchar* frNextXPtr = frame.ptr(y) + (x + dx) * 3;
			const uchar* frNextYPtr = frame.ptr(y + dy) + x * 3;
			const uchar* bgPtr = bg.ptr(y) + x * 3;
			const uchar* bgNextXPtr = bg.ptr(y) + (x + dx) * 3;
			const uchar* bgNextYPtr = bg.ptr(y + dy) + x * 3;

			float frVal = 0;
			float frNextXVal = 0;
			float frNextYVal = 0;
			float bgVal = 0;
			float bgNextXVal = 0;
			float bgNextYVal = 0;
			for (int c = 0; c < 3; ++c) {
				frVal += BGR2GRAY[c] * frPtr[c];
				frNextXVal += BGR2GRAY[c] * frNextXPtr[c];
				frNextYVal += BGR2GRAY[c] * frNextYPtr[c];
				bgVal += BGR2GRAY[c] * bgPtr[c];
				bgNextXVal += BGR2GRAY[c] * bgNextXPtr[c];
				bgNextYVal += BGR2GRAY[c] * bgNextYPtr[c];
			}

			float frGx = frNextXVal - frVal;
			float frGy = frVal - frNextYVal;
			float bgGx = bgNextXVal - bgVal;
			float bgGy = bgVal - bgNextYVal;

			float frG = std::sqrt(frGx * frGx + frGy * frGy);
			float bgG = std::sqrt(bgGx * bgGx + bgGy * bgGy);

			float* weightsPtr = weights.ptr<float>(y);
			weightsPtr[x] = (params.weightSmootTerm + bgG) / (params.weightSmootTerm + std::max(frG, bgG));
		}
	}
}

void PhysicalShadRem::getShadows(const cv::Mat& features, const cv::Mat& weights, const ConnCompGroup& candidates,
		cv::Mat& shadows) {
	if (gmm.empty()) {
		std::vector<double> initVars(3, params.gmmInitVar);
		std::vector<double> minVars(3, params.gmmInitVar);
		gmm.init(params.gmmGaussians, features.rows, features.cols, 3, initVars, minVars, params.gmmStdThreshold,
				params.gmmWinnerTakesAll, params.gmmLearningRate, params.gmmSortMode, params.gmmFitLogistic);
	}

	cv::Mat gmmWeights = weights.clone();
	if (!params.learnBorders) {
		cv::Mat borders(features.size(), CV_8U, cv::Scalar(0));
		candidateShadows.draw(borders, cv::Scalar(255), false);
		gmmWeights.setTo(cv::Scalar(0), borders);
	}

	gmm.update(features, gmmWeights);

	gmm.evaluate(features, gaussians, params.gmmAccumWeightThresh, candidates.mask);
	posteriors = gaussians.mul(weights);

	cv::Mat thresh = posteriors.clone();
	cv::threshold(thresh, thresh, params.postThresh, 255, cv::THRESH_BINARY);
	thresh.convertTo(shadows, CV_8U);
}
