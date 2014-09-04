// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef GEOMETRYSHADREM_H_
#define GEOMETRYSHADREM_H_

#include <cxcore.h>
#include "GeometryShadRemParams.h"

class ConnComp;

/**
 * Implemented from:
 *    Shadow elimination for effective moving object detection by Gaussian shadow modeling
 *    Hsieh et al. (IVC 2003)
 */
class GeometryShadRem {

	public:
		GeometryShadRem(const GeometryShadRemParams& params = GeometryShadRemParams());
		virtual ~GeometryShadRem();

		void removeShadows(const cv::Mat& frame, const cv::Mat& fg, const cv::Mat& bg, cv::Mat& srMask);

	private:
		GeometryShadRemParams params;

		cv::Mat grayFrame;

		void extractHeads(const cv::Mat& grayFrame, const ConnComp& cc, std::vector<int>& heads);
		void extractBoundaries(const cv::Mat& grayFrame, const ConnComp& cc, const std::vector<int>& heads,
				std::vector<std::pair<int, int> >& boundaries,
				bool& rightShadows);
		void extractSegments(const cv::Size& frameSize, const ConnComp& cc, const std::vector<int>& heads,
				const std::vector<std::pair<int, int> >& boundaries,
				bool rightShadows, std::vector<cv::Point>& segmentMaskPositions, std::vector<cv::Mat>& segmentMasks
				, double& splitAngle);
		void splitSegment(const std::map<int, int>& ccVerticalProjection,
		const cv::Point& segmentMaskPos, const cv::Mat& segmentMask, double splitAngle,
		cv::Mat& segmentSplitMask);
		void classifySegment(const cv::Mat& grayFrame, const cv::Mat& segmentMask, const cv::Mat& segmentSplitMask,
				cv::Mat& segmentShadowMask);
		void getVerticalIntensityProjection(const cv::Mat& grayFrame, const ConnComp& cc,
				std::map<int, int>& verticalIntensityProjection);
		void getVerticalEdgeProjection(const cv::Mat& grayFrame, const ConnComp& cc,
				std::map<int, int>& verticalEdgeProjection);
};

#endif /* GEOMETRYSHADREM_H_ */
