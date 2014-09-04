// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include <limits>
#include <cv.h>
#include "GeometryShadRem.h"
#include "utils/ConnCompGroup.h"

GeometryShadRem::GeometryShadRem(const GeometryShadRemParams& params) {
	this->params = params;
}

GeometryShadRem::~GeometryShadRem() {
}

void GeometryShadRem::removeShadows(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, cv::Mat& srMask) {
	ConnCompGroup fg(fgMask);
	fg.mask.copyTo(srMask);
	cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);

	for (int b = 0; b < (int) fg.comps.size(); ++b) {
		std::vector<int> heads;
		extractHeads(grayFrame, fg.comps[b], heads);
		std::vector<std::pair<int, int> > boundaries;
		bool rightShadows;
		extractBoundaries(grayFrame, fg.comps[b], heads, boundaries, rightShadows);
		std::vector<cv::Point> segmentMaskPositions;
		std::vector<cv::Mat> segmentMasks;
		double splitAngle;
		extractSegments(grayFrame.size(), fg.comps[b], heads, boundaries, rightShadows, segmentMaskPositions,
				segmentMasks, splitAngle);

		std::map<int, int> projection;
		fg.comps[b].verticalProjection(projection);
		for (int i = 0; i < (int) segmentMasks.size() && i < 9; ++i) {
			cv::Mat segmentSplitMask;
			splitSegment(projection, segmentMaskPositions[i], segmentMasks[i], splitAngle, segmentSplitMask);

			cv::Mat segmentShadowMask;
			classifySegment(grayFrame, segmentMasks[i], segmentSplitMask, segmentShadowMask);

			cv::Rect segmentBox(segmentMaskPositions[i], segmentMasks[i].size());
			cv::Mat srMaskRoi(srMask, segmentBox);
			srMaskRoi.setTo(0, segmentShadowMask);
		}
	}

	if (params.cleanSrMask) {
		ConnCompGroup ccg;
		ccg.update(srMask, true, true);
		ccg.mask.copyTo(srMask);
	}
}

void GeometryShadRem::extractHeads(const cv::Mat& grayFrame, const ConnComp& cc, std::vector<int>& heads) {
	heads.clear();

	std::map<int, int> projection;
	getVerticalIntensityProjection(grayFrame, cc, projection);
	int maxP = -1;
	for (std::map<int, int>::iterator iter = projection.begin(); iter != projection.end(); ++iter) {
		if (iter->second > maxP) {
			maxP = iter->second;
		}
	}

	int thresh = maxP / params.headThreshRatio;
	int seqSize = 0;
	int prevP = -1;
	for (std::map<int, int>::iterator iter = projection.begin(); iter != projection.end(); ++iter) {
		int p = iter->second;

		if (p > prevP && p >= thresh) {
			++seqSize;
			prevP = p;

			if (seqSize == params.minHeadSeq) {
				heads.push_back(iter->first);
			}
			else if (seqSize > params.minHeadSeq) {
				heads.back() = iter->first;
			}
		}
		else {
			seqSize = 0;
			prevP = -1;
		}
	}
}

void GeometryShadRem::extractBoundaries(const cv::Mat& grayFrame, const ConnComp& cc, const std::vector<int>& heads,
		std::vector<std::pair<int, int> >& boundaries,
		bool& rightShadows) {
	boundaries.clear();

	std::map<int, int> projection;
	getVerticalEdgeProjection(grayFrame, cc, projection);
	int maxP = -1;
	for (std::map<int, int>::iterator iter = projection.begin(); iter != projection.end(); ++iter) {
		if (iter->second > maxP) {
			maxP = iter->second;
		}
	}

	std::vector<int> edges;
	int thresh = maxP / params.edgeThreshRatio;
	int seqSize = 0;
	int prevP = -1;
	for (std::map<int, int>::iterator iter = projection.begin(); iter != projection.end(); ++iter) {
		int p = iter->second;

		if (p > prevP && p >= thresh) {
			++seqSize;
			prevP = p;

			if (seqSize == params.minEdgeSeq) {
				edges.push_back(iter->first);
			}
			else if (seqSize > params.minEdgeSeq) {
				edges.back() = iter->first;
			}
		}
		else {
			seqSize = 0;
			prevP = -1;
		}
	}

	int x2Prev = -1;
	for (std::vector<int>::const_iterator headIter = heads.begin(); headIter < heads.end(); ++headIter) {
		int headX = *headIter;

		if (headX > x2Prev) {
			int x1 = cc.box.x;
			int x2 = (cc.box.x + cc.box.width) - 1;
			for (std::vector<int>::iterator edgeIter = edges.begin(); edgeIter < edges.end(); ++edgeIter) {
				int edgeX = *edgeIter;

				if (edgeX <= headX) {
					x1 = edgeX;
				}

				if (edgeX > headX) {
					x2 = edgeX;
					break;
				}
			}

			boundaries.push_back(std::pair<int, int>(x1, x2));

			x2Prev = x2;
		}
	}

	if (!heads.empty()) {
		int leftGap = std::abs(cc.box.x - heads.front());
		int rightGap = std::abs(((cc.box.x + cc.box.width) - 1) - heads.back());
		rightShadows = (rightGap > leftGap);
	}
}

void GeometryShadRem::extractSegments(const cv::Size& frameSize, const ConnComp& cc, const std::vector<int>& heads,
		const std::vector<std::pair<int, int> >& boundaries,
		bool rightShadows, std::vector<cv::Point>& segmentMaskPositions,
		std::vector<cv::Mat>& segmentMasks, double& splitAngle) {
	segmentMaskPositions.clear();
	segmentMasks.clear();

	if (!boundaries.empty()) {
		std::vector<int> segments;
		if (rightShadows) {
			segments.push_back(cc.box.x);
			for (int i = 1; i < (int) boundaries.size(); ++i) {
				segments.push_back(boundaries[i].first);
			}
			segments.push_back(cc.box.x + cc.box.width);
		}
		else {
			segments.push_back(cc.box.x);
			for (int i = 0; i < (int) boundaries.size() - 1; ++i) {
				segments.push_back(boundaries[i].second + 1);
			}
			segments.push_back(cc.box.x + cc.box.width);
		}

		for (int i = 0; i < ((int) segments.size() - 1); ++i) {
			segmentMaskPositions.push_back(cv::Point(segments[i], cc.box.y));
			cv::Rect segmentBox;
			segmentBox.x = segments[i] - cc.box.x;
			segmentBox.y = 0;
			segmentBox.width = segments[i + 1] - segments[i];
			segmentBox.height = cc.box.height;
			segmentMasks.push_back(cc.boxMask(cv::Rect(segmentBox)));
		}
	}

	splitAngle = 0;
	if (!segmentMasks.empty()) {
		cv::Mat& segmentMask = (rightShadows ? segmentMasks.back() : segmentMasks.front());
		cv::Moments moments = cv::moments(segmentMask, true);
		splitAngle = std::atan2(2 * moments.mu11, moments.mu20 - moments.mu02) / 2;
		if (rightShadows && splitAngle > 0) {
			splitAngle -= CV_PI / 2;
		}
		else if (!rightShadows && splitAngle < 0) {
			splitAngle += CV_PI / 2;
		}
	}
}

void GeometryShadRem::splitSegment(const std::map<int, int>& ccVerticalProjection,
const cv::Point& segmentMaskPos, const cv::Mat& segmentMask, double splitAngle,
cv::Mat& segmentSplitMask) {
	segmentSplitMask = cv::Mat(segmentMask.size(), CV_8U, cv::Scalar(0));

	cv::Moments moments = cv::moments(segmentMask, true);
	int centerY = moments.m01 / moments.m00;

	int xBottomL = segmentMask.cols;
	int xBottomR = -1;
	for (int y = centerY + 1; y < segmentMask.rows; ++y) {
		const uchar* ptr = segmentMask.ptr(y);

		for (int xL = 0; xL < xBottomL; ++xL) {
			if (ptr[xL] > 0) {
				xBottomL = xL;
				break;
			}
		}

		for (int xR = segmentMask.cols - 1; xR > xBottomR; --xR) {
			if (ptr[xR] > 0) {
				xBottomR = xR;
				break;
			}
		}
	}

	int shift = std::max((xBottomR - xBottomL) / params.bottomShiftRatio, 1);
	xBottomL += shift;
	xBottomR -= shift;

	int splitX = -1;
	int maxDiff = 0;
	for (int x = xBottomL; x <= xBottomR; ++x) {
		int pL = ccVerticalProjection.find((x - 1) + segmentMaskPos.x)->second;
		int pR = ccVerticalProjection.find((x + 1) + segmentMaskPos.x)->second;
		int pDiff = std::abs(pR - pL);
		if (pDiff > maxDiff) {
			splitX = x;
			maxDiff = pDiff;
		}
	}

	if (splitX != -1) {
		int pX = ccVerticalProjection.find(splitX + segmentMaskPos.x)->second;
		int pL = ccVerticalProjection.find((splitX - 1) + segmentMaskPos.x)->second;
		int pR = ccVerticalProjection.find((splitX + 1) + segmentMaskPos.x)->second;

		if (pL < pX && pL < pR) {
			--splitX;
		}
		else if (pR < pX && pR < pL) {
			++splitX;
		}

		int splitY = 0;
		for (int y = segmentMask.rows - 1; y > 0; --y) {
			if (segmentMask.ptr(y)[splitX] > segmentMask.ptr(y - 1)[splitX]) {
				splitY = y;
				break;
			}
		}

		double m = std::tan(splitAngle);
		double c = splitY - m * splitX;

		cv::Mat r1(segmentMask.size(), CV_8U, cv::Scalar(0));
		cv::Mat r2(segmentMask.size(), CV_8U, cv::Scalar(0));
		for (int y = 0; y < segmentMask.rows; ++y) {
			const uchar* maskPtr = segmentMask.ptr(y);
			uchar* r1Ptr = r1.ptr(y);
			uchar* r2Ptr = r2.ptr(y);

			int xThresh = 0;
			if (std::abs(m) > std::numeric_limits<double>::max()) {
				xThresh = splitX;
			}
			else if (std::abs(m) > 0) {
				xThresh = (y - c) / m;
			}
			for (int x = 0; x < segmentMask.cols; ++x) {
				if (maskPtr[x] > 0) {
					bool r1Point;
					if (m == 0) {
						r1Point = (y < splitY);
					}
					else {
						r1Point = (x < xThresh);
					}

					if (r1Point) {
						r1Ptr[x] = 255;
					}
					else {
						r2Ptr[x] = 255;
					}
				}
			}
		}

		cv::Moments r1Moments = cv::moments(r1, true);
		double r1Angle = std::atan2(2 * r1Moments.mu11, r1Moments.mu20 - r1Moments.mu02) / 2;
		cv::Moments r2Moments = cv::moments(r2, true);
		double r2Angle = std::atan2(2 * r2Moments.mu11, r2Moments.mu20 - r2Moments.mu02) / 2;
		if (std::abs(r1Angle) < std::abs(r2Angle)) {
			r1.copyTo(segmentSplitMask);
		}
		else {
			r2.copyTo(segmentSplitMask);
		}
	}
}

void GeometryShadRem::classifySegment(const cv::Mat& grayFrame, const cv::Mat& segmentMask,
		const cv::Mat& segmentSplitMask, cv::Mat& segmentShadowMask) {
	segmentShadowMask = cv::Mat(segmentMask.size(), CV_8U, cv::Scalar(0));

	cv::Moments moments = cv::moments(segmentSplitMask, true);
	double n = moments.m00;
	double meanX = moments.m10 / n;
	double meanY = moments.m01 / n;
	double angle = std::atan2(2 * moments.mu11, moments.mu20 - moments.mu02) / 2;

	// calculate statistics
	double sSum = 0;
	double sSqrSum = 0;
	double tSum = 0;
	double tSqrSum = 0;
	double gSum = 0;
	double gSqrSum = 0;
	for (int y = 0; y < segmentSplitMask.rows; ++y) {
		const uchar* splitPtr = segmentSplitMask.ptr(y);
		const uchar* framePtr = grayFrame.ptr(y);

		for (int x = 0; x < segmentSplitMask.cols; ++x) {
			if (splitPtr[x] > 0) {
				double s = std::cos(angle) * (x - meanX) - std::sin(angle) * (y - meanY);
				double t = std::sin(angle) * (x - meanX) + std::cos(angle) * (y - meanY);
				int g = framePtr[x];

				sSum += s;
				sSqrSum += s * s;
				tSum += t;
				tSqrSum += t * t;
				gSum += g;
				gSqrSum += g * g;
			}
		}
	}
	double sMean = sSum / n;
	double sVar = (sSqrSum - sSum * sMean) / (n - 1);
	double tMean = tSum / n;
	double tVar = (tSqrSum - tSum * tMean) / (n - 1);
	double gMean = gSum / n;
	double gVar = (gSqrSum - gSum * gMean) / (n - 1);

	// calculate threshold
	double sWeight = (1 - params.gWeight) * params.sRelativeWeight;
	double tWeight = 1 - (params.gWeight + sWeight);
	double thresh = 0;
	for (int y = 0; y < segmentSplitMask.rows; ++y) {
		const uchar* splitPtr = segmentSplitMask.ptr(y);
		const uchar* framePtr = grayFrame.ptr(y);

		for (int x = 0; x < segmentSplitMask.cols; ++x) {
			if (splitPtr[x] > 0) {
				double s = std::cos(angle) * (x - meanX) - std::sin(angle) * (y - meanY);
				double t = std::sin(angle) * (x - meanX) + std::cos(angle) * (y - meanY);
				int g = framePtr[x];

				double sTerm = (sWeight * s * s) / sVar;
				double tTerm = (tWeight * t * t) / tVar;
				double gTerm = (params.gWeight * (g - gMean) * (g - gMean)) / gVar;
				thresh += std::exp(-(sTerm + tTerm + gTerm));
			}
		}
	}
	thresh /= n;
	thresh *= params.thresholdScale;

	// classify shadow pixels
	for (int y = 0; y < segmentMask.rows; ++y) {
		const uchar* maskPtr = segmentMask.ptr(y);
		const uchar* framePtr = grayFrame.ptr(y);
		uchar* shadowMaskPtr = segmentShadowMask.ptr(y);

		for (int x = 0; x < segmentMask.cols; ++x) {
			if (maskPtr[x] > 0) {
				double s = std::cos(angle) * (x - meanX) - std::sin(angle) * (y - meanY);
				double t = std::sin(angle) * (x - meanX) + std::cos(angle) * (y - meanY);
				int g = framePtr[x];

				double sTerm = (sWeight * s * s) / sVar;
				double tTerm = (tWeight * t * t) / tVar;
				double gTerm = (params.gWeight * (g - gMean) * (g - gMean)) / gVar;
				double val = std::exp(-(sTerm + tTerm + gTerm));
				if (val > thresh) {
					shadowMaskPtr[x] = 255;
				}
			}
		}
	}
}

void GeometryShadRem::getVerticalIntensityProjection(const cv::Mat& grayFrame, const ConnComp& cc,
		std::map<int, int>& verticalIntensityProjection) {
	verticalIntensityProjection.clear();

	for (int i = 0; i < (int) cc.pixels.size(); ++i) {
		const uchar* ptr = grayFrame.ptr(cc.pixels[i].y);
		verticalIntensityProjection[cc.pixels[i].x] += ptr[cc.pixels[i].x];
	}

	if (params.smoothFactor > 0 && 2 * params.smoothFactor < (int) verticalIntensityProjection.size()) {
		int minX = verticalIntensityProjection.begin()->first;
		std::vector<int> tmp;
		for (std::map<int, int>::iterator iter = verticalIntensityProjection.begin();
				iter != verticalIntensityProjection.end(); ++iter) {
			tmp.push_back(iter->second);
		}
		verticalIntensityProjection.clear();

		int pStart = 0;
		int pSum = 0;
		for (int count = 0; count < (int) tmp.size(); ++count) {
			int p = tmp[count];

			if (count < 2 * params.smoothFactor) {
				pSum += p;
			}
			else {
				int x = (minX + count) - params.smoothFactor;
				pSum = pSum + p - pStart;
				pStart = tmp[count - 2 * params.smoothFactor];

				verticalIntensityProjection[x] = pSum / (2 * params.smoothFactor + 1);
			}
		}
	}
}

void GeometryShadRem::getVerticalEdgeProjection(const cv::Mat& grayFrame, const ConnComp& cc,
		std::map<int, int>& verticalEdgeProjection) {
	verticalEdgeProjection.clear();

	for (int i = 0; i < (int) cc.pixels.size(); ++i) {
		if (cc.pixels[i].x - params.maxEdgeDistance >= 0 && cc.pixels[i].x + params.maxEdgeDistance < grayFrame.cols) {
			int edgeSum = 0;
			const uchar* ptr = grayFrame.ptr(cc.pixels[i].y);
			for (int dx = 1; dx <= params.maxEdgeDistance; ++dx) {
				edgeSum += std::abs(ptr[cc.pixels[i].x + dx] - ptr[cc.pixels[i].x - dx]);
			}

			verticalEdgeProjection[cc.pixels[i].x] += edgeSum;
		}
	}
}
