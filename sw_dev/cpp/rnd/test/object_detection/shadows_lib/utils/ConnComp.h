// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef CONNCOMP_H_
#define CONNCOMP_H_

#include <map>
#include <cxcore.h>

/**
 * Stores all the necessary information for a connected component.
 */
class ConnComp {

	public:
		cv::Point center;
		cv::Rect box;
		cv::Mat boxMask;
		std::vector<cv::Point> pixels;
		std::vector<std::vector<cv::Point> > contours;

		ConnComp();
		virtual ~ConnComp();

		void
		draw(cv::Mat& dst, const cv::Scalar& color = cv::Scalar(255, 255, 255), bool filled = true) const;
		void verticalProjection(std::map<int, int>& verticalProjection) const;
};

#endif /* CONNCOMP_H_ */
