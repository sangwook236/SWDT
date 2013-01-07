//#include "stdafx.h"
#include <opencv/cxcore.h>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cmath>

#define __COLUMN_MAJOR 1


namespace opencv {

void pca()
{
#if defined(__COLUMN_MAJOR)
	const size_t ROW_SIZE = 9;
	const size_t COL_SIZE = 4;

	const size_t MIN_SIZE = std::min(ROW_SIZE, COL_SIZE);

	//
	CvMat *inputMat = cvCreateMat(ROW_SIZE, COL_SIZE, CV_32FC1);
	cvmSet(inputMat, 0, 0, 225);
	cvmSet(inputMat, 1, 0, 229);
	cvmSet(inputMat, 2, 0, 48);
	cvmSet(inputMat, 3, 0, 251);
	cvmSet(inputMat, 4, 0, 33);
	cvmSet(inputMat, 5, 0, 238);
	cvmSet(inputMat, 6, 0, 0);
	cvmSet(inputMat, 7, 0, 255);
	cvmSet(inputMat, 8, 0, 217);
	cvmSet(inputMat, 0, 1, 10);
	cvmSet(inputMat, 1, 1, 219);
	cvmSet(inputMat, 2, 1, 24);
	cvmSet(inputMat, 3, 1, 255);
	cvmSet(inputMat, 4, 1, 18);
	cvmSet(inputMat, 5, 1, 247);
	cvmSet(inputMat, 6, 1, 17);
	cvmSet(inputMat, 7, 1, 255);
	cvmSet(inputMat, 8, 1, 2);
	cvmSet(inputMat, 0, 2, 196);
	cvmSet(inputMat, 1, 2, 35);
	cvmSet(inputMat, 2, 2, 234);
	cvmSet(inputMat, 3, 2, 232);
	cvmSet(inputMat, 4, 2, 59);
	cvmSet(inputMat, 5, 2, 244);
	cvmSet(inputMat, 6, 2, 243);
	cvmSet(inputMat, 7, 2, 57);
	cvmSet(inputMat, 8, 2, 226);
	cvmSet(inputMat, 0, 3, 255);
	cvmSet(inputMat, 1, 3, 223);
	cvmSet(inputMat, 2, 3, 224);
	cvmSet(inputMat, 3, 3, 255);
	cvmSet(inputMat, 4, 3, 0);
	cvmSet(inputMat, 5, 3, 255);
	cvmSet(inputMat, 6, 3, 249);
	cvmSet(inputMat, 7, 3, 255);
	cvmSet(inputMat, 8, 3, 235);

	CvMat *sampleMat = cvCreateMat(ROW_SIZE, 1, CV_32FC1);
	cvmSet(sampleMat, 0, 0, 20);
	cvmSet(sampleMat, 1, 0, 244);
	cvmSet(sampleMat, 2, 0, 44);
	cvmSet(sampleMat, 3, 0, 246);
	cvmSet(sampleMat, 4, 0, 21);
	cvmSet(sampleMat, 5, 0, 244);
	cvmSet(sampleMat, 6, 0, 4);
	cvmSet(sampleMat, 7, 0, 255);
	cvmSet(sampleMat, 8, 0, 2);
#else
	const size_t ROW_SIZE = 4;
	const size_t COL_SIZE = 9;

	const size_t MIN_SIZE = std::min(ROW_SIZE, COL_SIZE);

	//
	CvMat *inputMat = cvCreateMat(ROW_SIZE, COL_SIZE, CV_32FC1);
	cvmSet(inputMat, 0, 0, 225);
	cvmSet(inputMat, 0, 1, 229);
	cvmSet(inputMat, 0, 2, 48);
	cvmSet(inputMat, 0, 3, 251);
	cvmSet(inputMat, 0, 4, 33);
	cvmSet(inputMat, 0, 5, 238);
	cvmSet(inputMat, 0, 6, 0);
	cvmSet(inputMat, 0, 7, 255);
	cvmSet(inputMat, 0, 8, 217);
	cvmSet(inputMat, 1, 0, 10);
	cvmSet(inputMat, 1, 1, 219);
	cvmSet(inputMat, 1, 2, 24);
	cvmSet(inputMat, 1, 3, 255);
	cvmSet(inputMat, 1, 4, 18);
	cvmSet(inputMat, 1, 5, 247);
	cvmSet(inputMat, 1, 6, 17);
	cvmSet(inputMat, 1, 7, 255);
	cvmSet(inputMat, 1, 8, 2);
	cvmSet(inputMat, 2, 0, 196);
	cvmSet(inputMat, 2, 1, 35);
	cvmSet(inputMat, 2, 2, 234);
	cvmSet(inputMat, 2, 3, 232);
	cvmSet(inputMat, 2, 4, 59);
	cvmSet(inputMat, 2, 5, 244);
	cvmSet(inputMat, 2, 6, 243);
	cvmSet(inputMat, 2, 7, 57);
	cvmSet(inputMat, 2, 8, 226);
	cvmSet(inputMat, 3, 0, 255);
	cvmSet(inputMat, 3, 1, 223);
	cvmSet(inputMat, 3, 2, 224);
	cvmSet(inputMat, 3, 3, 255);
	cvmSet(inputMat, 3, 4, 0);
	cvmSet(inputMat, 3, 5, 255);
	cvmSet(inputMat, 3, 6, 249);
	cvmSet(inputMat, 3, 7, 255);
	cvmSet(inputMat, 3, 8, 235);

	CvMat *sampleMat = cvCreateMat(1, COL_SIZE, CV_32FC1);
	cvmSet(sampleMat, 0, 0, 20);
	cvmSet(sampleMat, 0, 1, 244);
	cvmSet(sampleMat, 0, 2, 44);
	cvmSet(sampleMat, 0, 3, 246);
	cvmSet(sampleMat, 0, 4, 21);
	cvmSet(sampleMat, 0, 5, 244);
	cvmSet(sampleMat, 0, 6, 4);
	cvmSet(sampleMat, 0, 7, 255);
	cvmSet(sampleMat, 0, 8, 2);
#endif

	//
#if defined(__COLUMN_MAJOR)
	CvMat *aveVecs = cvCreateMat(ROW_SIZE, 1, CV_32FC1);

	CvMat *eigVals = cvCreateMat((int)MIN_SIZE, 1, CV_32FC1);
	// row-major
	CvMat *eigVecs = cvCreateMat((int)MIN_SIZE, ROW_SIZE, CV_32FC1);

	//
	cvCalcPCA(inputMat, aveVecs, eigVals, eigVecs, CV_PCA_DATA_AS_COL);
#else
	CvMat *aveVecs = cvCreateMat(1, COL_SIZE, CV_32FC1);

	CvMat *eigVals = cvCreateMat(1, (int)MIN_SIZE, CV_32FC1);
	// row-major
	CvMat *eigVecs = cvCreateMat((int)MIN_SIZE, (int)COL_SIZE, CV_32FC1);

	//
	cvCalcPCA(inputMat, aveVecs, eigVals, eigVecs, CV_PCA_DATA_AS_ROW);
#endif

#if defined(__COLUMN_MAJOR)
	std::vector<double> aveVecs_tmp;
	for (int i = 0; i < aveVecs->rows; ++i)
		aveVecs_tmp.push_back(cvmGet(aveVecs, i, 0));
	std::vector<double> eigVals_tmp;
	for (int i = 0; i < eigVals->rows; ++i)
		eigVals_tmp.push_back(cvmGet(eigVals, i, 0));
#else
	std::vector<double> aveVecs_tmp;
	for (int i = 0; i < aveVecs->cols; ++i)
		aveVecs_tmp.push_back(cvmGet(aveVecs, 0, i));
	std::vector<double> eigVals_tmp;
	for (int i = 0; i < eigVals->cols; ++i)
		eigVals_tmp.push_back(cvmGet(eigVals, 0, i));
#endif
	std::vector<double> eigVecs_tmp;
	for (int i = 0; i < eigVecs->rows; ++i)
		for (int j = 0; j < eigVecs->cols; ++j)
			eigVecs_tmp.push_back(cvmGet(eigVecs, i, j));

	//
	const size_t MAT_RANK = std::min(eigVecs->rows, eigVecs->cols);

	const size_t PRINCIPAL_COMPONENT_NUM = 3;
	assert(PRINCIPAL_COMPONENT_NUM <= MAT_RANK);

#if defined(__COLUMN_MAJOR)
	// row-major
	CvMat *pcVecs = cvCreateMat(PRINCIPAL_COMPONENT_NUM, ROW_SIZE, CV_32FC1);
#else
	// row-major
	CvMat *pcVecs = cvCreateMat(PRINCIPAL_COMPONENT_NUM, COL_SIZE, CV_32FC1);
#endif
	std::vector<double> pcVecs_tmp;
	for (int i = 0; i < pcVecs->rows; ++i)
		for (int j = 0; j < pcVecs->cols; ++j)
		{
			cvmSet(pcVecs, i, j, cvmGet(eigVecs, i, j));
			pcVecs_tmp.push_back(cvmGet(pcVecs, i, j));
		}

	//
#if defined(__COLUMN_MAJOR)
	CvMat *inputMat_tilde = cvCreateMat(COL_SIZE, pcVecs->rows, CV_32FC1);  // caution: matrix size
#else
	CvMat *inputMat_tilde = cvCreateMat(ROW_SIZE, pcVecs->rows, CV_32FC1);  // caution: matrix size
#endif

	cvProjectPCA(inputMat, aveVecs, pcVecs, inputMat_tilde);

	std::vector<double> inputMat_tilde_tmp;
	for (int i = 0; i < inputMat_tilde->rows; ++i)
		for (int j = 0; j < inputMat_tilde->cols; ++j)
			inputMat_tilde_tmp.push_back(cvmGet(inputMat_tilde, i, j));

	//
	CvMat *sampleMat_tilde = cvCreateMat(1, pcVecs->rows, CV_32FC1);

	cvProjectPCA(sampleMat, aveVecs, pcVecs, sampleMat_tilde);

	std::vector<double> sampleMat_tilde_tmp;
	for (int i = 0; i < sampleMat_tilde->cols; ++i)
		sampleMat_tilde_tmp.push_back(cvmGet(sampleMat_tilde, 0, i));

	//
	cvReleaseMat(&inputMat);
	cvReleaseMat(&sampleMat);
	cvReleaseMat(&aveVecs);
	cvReleaseMat(&eigVals);
	cvReleaseMat(&eigVecs);
	cvReleaseMat(&pcVecs);
	cvReleaseMat(&inputMat_tilde);
	cvReleaseMat(&sampleMat_tilde);
}

}  // namespace opencv
