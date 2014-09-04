// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef GABORFILTER_H_
#define GABORFILTER_H_

#include <cxcore.h>

/**
 * GaborFilter class.
 */
class GaborFilter {

	public:
		GaborFilter();
		virtual ~GaborFilter();

		static void getDistance(const std::vector<cv::Mat>& projectionsA,
				const std::vector<cv::Mat>& projectionsB, cv::Mat& distance, const cv::Mat& mask =
						cv::Mat(), bool normalize = true);

		void createKernels(int kernelRadius, float wavelength, float aspectRatio,
				const std::vector<float>& bandwidths, const std::vector<float>& orientations,
				const std::vector<float>& phases);

		void filter(const cv::Mat& grayFrame, std::vector<cv::Mat>& projections,
				const cv::Mat& mask = cv::Mat(), int neighborhood = 1, bool normalize = true) const;

	private:
		int kernelRadius;
		std::vector<cv::Mat> kernels;

		static void getIntegral(const cv::Mat& frame, cv::Mat& integral);
};

#endif /* GABORFILTER_H_ */
