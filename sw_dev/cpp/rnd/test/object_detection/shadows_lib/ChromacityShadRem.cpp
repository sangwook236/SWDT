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
#include "ChromacityShadRem.h"
#include "utils/ConnCompGroup.h"

ChromacityShadRem::ChromacityShadRem(const ChromacityShadRemParams& params) {
	this->params = params;
}

ChromacityShadRem::~ChromacityShadRem() {
}

void ChromacityShadRem::removeShadows(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, cv::Mat& srMask) {
	ConnCompGroup fg(fgMask);
	fgMask.copyTo(srMask);

	ConnCompGroup darkPixels;
	ConnCompGroup shadows;
	cv::Mat hsvFrame, hsvBg;
	cv::cvtColor(frame, hsvFrame, CV_BGR2HSV);
	cv::cvtColor(bg, hsvBg, CV_BGR2HSV);

	extractDarkPixels(hsvFrame, fg, hsvBg, darkPixels);
	extractShadows(hsvFrame, darkPixels, hsvBg, shadows);

	srMask.setTo(0, shadows.mask);

	if (params.cleanSrMask) {
		ConnCompGroup ccg;
		ccg.update(srMask, true, true);
		ccg.mask.copyTo(srMask);
	}
}

void ChromacityShadRem::extractDarkPixels(const cv::Mat& hsvFrame, const ConnCompGroup& fg, const cv::Mat& hsvBg,
		ConnCompGroup& darkPixels) {
	cv::Mat mask(hsvFrame.size(), CV_8U, cv::Scalar(0));

	for (int cc = 0; cc < (int) fg.comps.size(); ++cc) {
		const ConnComp& object = fg.comps[cc];

		for (int p = 0; p < (int) object.pixels.size(); ++p) {
			int x = object.pixels[p].x;
			int y = object.pixels[p].y;

			const uchar* hsvFramePtr = hsvFrame.ptr(y) + x * 3;
			const uchar* hsvBgPtr = hsvBg.ptr(y) + x * 3;

			float vRatio = (float) hsvFramePtr[2] / hsvBgPtr[2];
			if (vRatio > params.vThreshLower && vRatio < params.vThreshUpper) {
				uchar* maskPtr = mask.ptr(y);
				maskPtr[x] = 255;
			}
		}
	}

	darkPixels.update(mask);
}

void ChromacityShadRem::extractShadows(const cv::Mat& hsvFrame, const ConnCompGroup& darkPixels,
		const cv::Mat& hsvBg, ConnCompGroup& shadows) {
	cv::Mat mask(hsvFrame.size(), CV_8U, cv::Scalar(0));

	for (int cc = 0; cc < (int) darkPixels.comps.size(); ++cc) {
		const ConnComp& object = darkPixels.comps[cc];

		for (int p = 0; p < (int) object.pixels.size(); ++p) {
			int x = object.pixels[p].x;
			int y = object.pixels[p].y;

			int hDiffSum = 0;
			int sDiffSum = 0;
			int winArea = 0;
			int minY = std::max(y - params.winSize, 0);
			int maxY = std::min(y + params.winSize, hsvFrame.rows - 1);
			int minX = std::max(x - params.winSize, 0);
			int maxX = std::min(x + params.winSize, hsvFrame.cols - 1);
			for (int i = minY; i <= maxY; ++i) {
				const uchar* hsvFramePtr = hsvFrame.ptr(i);
				const uchar* hsvBgPtr = hsvBg.ptr(i);

				for (int j = minX; j <= maxX; ++j) {
					int hDiff = CV_IABS(hsvFramePtr[j * 3] - hsvBgPtr[j * 3]);
					if (hDiff > 90) {
						hDiff = 180 - hDiff;
					}
					hDiffSum += hDiff;

					int sDiff = hsvFramePtr[j * 3 + 1] - hsvBgPtr[j * 3 + 1];
					sDiffSum += sDiff;

					++winArea;
				}
			}

			bool hIsShadow = (hDiffSum / winArea < params.hThresh);
			bool sIsShadow = (sDiffSum / winArea < params.sThresh);

			if (hIsShadow && sIsShadow) {
				uchar* maskPtr = mask.ptr(y);
				maskPtr[x] = 255;
			}
		}
	}

	shadows.update(mask);
}
