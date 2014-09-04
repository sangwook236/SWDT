// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef CONNCOMPGROUP_H_
#define CONNCOMPGROUP_H_

#include "ConnComp.h"

/**
 * Detects and stores a sequence of connected components found in some
 * foreground mask.
 */
class ConnCompGroup {

	public:
		cv::Mat mask;
		std::vector<ConnComp> comps;

		ConnCompGroup(const cv::Mat& fgMask = cv::Mat());
		virtual ~ConnCompGroup();

		void update(const cv::Mat& fgMask, bool clean = false, bool fill = false, int minPerim = 0);
		void draw(cv::Mat& dst, const cv::Scalar& color = cv::Scalar(255, 255, 255), bool filled = true) const;

	private:
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
};

#endif /* CONNCOMPGROUP_H_ */
