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
#include "GaborFilter.h"

GaborFilter::GaborFilter() {
}

GaborFilter::~GaborFilter() {
}

void GaborFilter::getDistance(const std::vector<cv::Mat>& projectionsA,
		const std::vector<cv::Mat>& projectionsB, cv::Mat& distance, const cv::Mat& mask,
		bool normalize) {
	int dimensions = std::min(projectionsA.size(), projectionsB.size());
	if (dimensions > 0) {
		int rows = projectionsA[0].rows;
		int cols = projectionsA[0].cols;

		distance.create(rows, cols, CV_32F);
		//--S [] 2014/08/28: Sang-Wook Lee
		//const float* projAPtrs[dimensions];
		//const float* projBPtrs[dimensions];
		std::vector<const float*> projAPtrs(dimensions, NULL);
		std::vector<const float*> projBPtrs(dimensions, NULL);
		//--E [] 2014/08/28: Sang-Wook Lee

		for (int y = 0; y < rows; ++y) {
			const uchar* maskPtr = (!mask.empty() ? mask.ptr(y) : NULL);
			float* distPtr = distance.ptr<float> (y);
			for (int d = 0; d < dimensions; ++d) {
				projAPtrs[d] = projectionsA[d].ptr<float> (y);
				projBPtrs[d] = projectionsB[d].ptr<float> (y);
			}

			for (int x = 0; x < cols; ++x) {
				float dist = 0;

				if (!maskPtr || maskPtr[x] > 0) {
					for (int d = 0; d < dimensions; ++d) {
						float diff = projBPtrs[d][x] - projAPtrs[d][x];
						dist += diff * diff;
					}
					dist = (normalize ? sqrt(dist / dimensions) : sqrt(dist));
				}

				distPtr[x] = dist;
			}
		}
	}
	else {
		distance.release();
	}
}

void GaborFilter::createKernels(int kernelRadius, float wavelength, float aspectRatio,
		const std::vector<float>& bandwidths, const std::vector<float>& orientations,
		const std::vector<float>& phases) {
	this->kernelRadius = kernelRadius;

	std::vector<float> sigmas;
	for (int b = 0; b < (int) bandwidths.size(); ++b) {
		float sigma = (wavelength / CV_PI) * sqrt(log(2.) / 2) * ((pow(2, bandwidths[b]) + 1)
				/ (pow(2, bandwidths[b]) - 1));
		sigmas.push_back(sigma);
	}

	kernels.resize(orientations.size() * sigmas.size() * phases.size());
	cv::Mat kernel;
	cv::Size kernelSize(2 * kernelRadius + 1, 2 * kernelRadius + 1);

	int count = 0;
	for (int s = 0; s < (int) sigmas.size(); ++s) {
		int radius = sigmas[s];
		float variance = sigmas[s] * sigmas[s];

		kernel.create(radius * 2 + 1, radius * 2 + 1, CV_32F);

		for (int o = 0; o < (int) orientations.size(); ++o) {
			for (int p = 0; p < (int) phases.size(); ++p) {
				for (int y = -radius; y <= radius; ++y) {
					float* kernelPtr = kernel.ptr<float> (radius - y);

					for (int x = -radius; x <= radius; ++x) {
						float xRot = x * cos(orientations[o]) + y * sin(orientations[o]);
						float yRot = -x * sin(orientations[o]) + y * cos(orientations[o]);

						float gaussian = exp(-(xRot * xRot + aspectRatio * aspectRatio * yRot
								* yRot) / (2 * variance));
						float sinusoid = cos(2 * CV_PI * (xRot / wavelength) + phases[p]);

						kernelPtr[x + radius] = gaussian * sinusoid;
					}
				}

				cv::resize(kernel, kernels[count], kernelSize);

				++count;
			}
		}
	}
}

void GaborFilter::filter(const cv::Mat& grayFrame, std::vector<cv::Mat>& projections,
		const cv::Mat& mask, int neighborhood, bool normalize) const {
	cv::Mat dilatedMask;
	if (!mask.empty() && neighborhood > 1) {
		cv::dilate(mask, dilatedMask, cv::Mat(neighborhood, neighborhood, CV_8U, cv::Scalar(255)));
	}
	else {
		dilatedMask = mask;
	}

	int dimensions = kernels.size();
	projections.resize(dimensions);
	for (int k = 0; k < dimensions; ++k) {
		projections[k].create(grayFrame.size(), CV_32F);
		projections[k].setTo(cv::Scalar(0));

		for (int y = 0; y < grayFrame.rows; ++y) {
			const uchar* maskPtr = (!dilatedMask.empty() ? dilatedMask.ptr(y) : NULL);
			float* projPtr = projections[k].ptr<float> (y);

			for (int x = 0; x < grayFrame.cols; ++x) {
				float val = 0;

				if (!maskPtr || maskPtr[x] > 0) {
					for (int dy = -kernelRadius; dy <= kernelRadius; ++dy) {
						int fy = y + dy;

						if (fy >= 0 && fy < grayFrame.rows) {
							const uchar* fPtr = grayFrame.ptr(fy);
							const float* kPtr = kernels[k].ptr<float> (dy + kernelRadius);

							for (int dx = -kernelRadius; dx <= kernelRadius; ++dx) {
								int fx = x + dx;

								if (fx >= 0 && fx < grayFrame.cols) {
									val += fPtr[fx] * kPtr[dx + kernelRadius];
								}
							}
						}
					}
				}

				projPtr[x] = val;
			}
		}

		if (neighborhood > 1) {
			cv::Mat integral;
			getIntegral(projections[k], integral);

			for (int y = 0; y < grayFrame.rows; ++y) {
				const uchar* maskPtr = (!mask.empty() ? mask.ptr(y) : NULL);
				float* projPtr = projections[k].ptr<float> (y);

				int y1 = std::max(y - (neighborhood - 1) / 2, 0);
				int y2 = std::min(y + neighborhood / 2, grayFrame.rows - 1);
				float* integralPtr1 = (y1 > 0 ? integral.ptr<float> (y1) : NULL);
				float* integralPtr2 = integral.ptr<float> (y2);

				for (int x = 0; x < grayFrame.cols; ++x) {
					float val = 0;

					if (!maskPtr || maskPtr[x] > 0) {
						int x1 = std::max(x - (neighborhood - 1) / 2, 0);
						int x2 = std::min(x + neighborhood / 2, grayFrame.cols - 1);

						val += integralPtr2[x2];
						if (integralPtr1) {
							val -= integralPtr1[x2];
						}
						if (x1 > 0) {
							val -= integralPtr2[x1 - 1];
						}
						if (integralPtr1 && x1 > 0) {
							val += integralPtr1[x1 - 1];
						}

						val /= ((x2 - x1) + 1) * ((y2 - y1) + 1);
					}

					projPtr[x] = val;
				}
			}
		}

		if (normalize) {
			cv::normalize(projections[k], projections[k], 0, 1, cv::NORM_MINMAX, -1, mask);
		}
	}
}

void GaborFilter::getIntegral(const cv::Mat& matrix, cv::Mat& integral) {
	integral.create(matrix.size(), CV_32F);

	cv::Mat rowSum(matrix.size(), CV_32F);
	float* prevRowSumPtr = NULL;
	for (int y = 0; y < matrix.rows; ++y) {
		const float* matrixPtr = matrix.ptr<float> (y);
		float* rowSumPtr = rowSum.ptr<float> (y);
		float* integralPtr = integral.ptr<float> (y);

		for (int x = 0; x < matrix.cols; ++x) {
			rowSumPtr[x] = matrixPtr[x];

			if (prevRowSumPtr) {
				rowSumPtr[x] += prevRowSumPtr[x];
			}

			integralPtr[x] = rowSumPtr[x];

			if (x > 0) {
				integralPtr[x] += integralPtr[x - 1];
			}
		}

		prevRowSumPtr = rowSumPtr;
	}
}
