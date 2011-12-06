#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <iostream>
#include <cassert>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

void print_opencv_matrix(const CvMat* mat);

void intrinsic_camera_params(const bool isPlanarCalibrationRigs);
void extrinsic_camera_params();
void fundamental_matrix();
void camera_matrix(const CvMat* fundamental_matrix, CvMat* P, CvMat* P_prime);
void fundamental_matrix(const CvMat* P, const CvMat* P_prime, CvMat* fundamental_matrix);


void camera_geometry()
{
	intrinsic_camera_params(false);  // camera calibration matrix
	std::cout << std::endl;
	extrinsic_camera_params();  // essential matrix
	std::cout << std::endl;
	fundamental_matrix();
}

void essential_matrix(const CvMat* rotation_vector, const CvMat* translation_vector, CvMat* essential_matrix)
{
	CvMat* skewMat = cvCreateMat(3, 3, CV_64FC1);
	cvSetZero(skewMat);
	cvmSet(skewMat, 0, 1, -cvmGet(translation_vector, 0, 2));
	cvmSet(skewMat, 0, 2, cvmGet(translation_vector, 0, 1));
	cvmSet(skewMat, 1, 0, cvmGet(translation_vector, 0, 2));
	cvmSet(skewMat, 1, 2, -cvmGet(translation_vector, 0, 0));
	cvmSet(skewMat, 2, 0, -cvmGet(translation_vector, 0, 1));
	cvmSet(skewMat, 2, 1, cvmGet(translation_vector, 0, 0));

	CvMat* rotMat = cvCreateMat(3, 3, CV_64FC1);
	cvRodrigues2(rotation_vector, rotMat);

	//cvGEMM(skewMat, rotMat, 1.0, NULL, 0.0, essential_matrix, 0);
	cvMatMul(skewMat, rotMat, essential_matrix);

	cvReleaseMat(&skewMat);
	cvReleaseMat(&rotMat);
}

void intrinsic_camera_params(const bool isPlanarCalibrationRigs)
{
	const int correspondenceCount = 12;
	const int imageCount = 2;
	const int imgWidth = 1600, imgHeight = 1200;
/*
	const double worldPts[] = {
		// for image 1                                                                                                            // for image 2
		-6375.79, -7292.85, -8963.5, -9109.77, -8339.89, -6130.0, -6528.26, -22746.9, -8442.19, -7879.51, -6835.6, -7889.53,      -6375.79, -7292.85, -8963.5, -9109.77, -8339.89, -6130.0, -6528.26, -22746.9, -8442.19, -7879.51, -6835.6, -7889.53,
		-14763.6, -16615.2, -19982.2, -20267.9, -18716.2, -14246.9, -15043.4, -50051.5, -19103.9, -18188.7, -15952.3, -18152.4,   -14763.6, -16615.2, -19982.2, -20267.9, -18716.2, -14246.9, -15043.4, -50051.5, -19103.9, -18188.7, -15952.3, -18152.4,
		-121.166, -136.389, -164.888, -166.048, -152.894, -115.55, -122.866, -414.82, -157.763, -149.995, -131.236, -150.07,      -121.166, -136.389, -164.888, -166.048, -152.894, -115.55, -122.866, -414.82, -157.763, -149.995, -131.236, -150.07,
	};
*/
	// normalized 8-point algorithm
	const double worldPts[] = {
		// for image 1                                                                                                   // for image 2
		-2661.08, 4239.22, 1394.28, 1554.19, 2078.23, -1724.04, -4194.6, 565.549, 1319.88, 1159.02, 5971.86, 1161.29,    -2661.08, 4239.22, 1394.28, 1554.19, 2078.23, -1724.04, -4194.6, 565.549, 1319.88, 1159.02, 5971.86, 1161.29,
		-2484.6, 2833.56, 453.964, 871.437, 1461.99, -2552.14, -4379.67, 62.372, 442.012, 650.432, 5959.46, 414.166,     -2484.6, 2833.56, 453.964, 871.437, 1461.99, -2552.14, -4379.67, 62.372, 442.012, 650.432, 5959.46, 414.166,
		-4.97828, 5.37613, 1.34403, 1.27898, 1.83671, -2.43475, -5.94024, 0.492512, 1.6971, 2.37888, 16.490, 2.35034,    -4.97828, 5.37613, 1.34403, 1.27898, 1.83671, -2.43475, -5.94024, 0.492512, 1.6971, 2.37888, 16.490, 2.35034,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,                                                                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	};
	const double imagePts[] = {
		// for image 1                                                    // for image 2
		535, 789, 1037, 1215, 1131, 709, 707, 1150, 778, 487, 365, 495,   26, 232, 457, 551, 468, 78, 113, 1143, 367, 239, 105, 257,
		500, 528, 337, 681, 795, 1050, 739, 130, 261, 273, 367, 178,      460, 511, 358, 686, 783, 982, 702, 57, 233, 175, 258, 76,
	};

	// method #1
	CvMat* objPts = cvCreateMat(correspondenceCount * imageCount, 3, CV_64FC1);
	CvMat* imgPts = cvCreateMat(correspondenceCount * imageCount, 2, CV_64FC1);
	//const CvMat objPts_arr = cvMat(3, correspondenceCount * imageCount, CV_64FC1, (void*)worldPts);
	//const CvMat imgPts_arr = cvMat(2, correspondenceCount * imageCount, CV_64FC1, (void*)imagePts);
	//cvTranspose(&objPts_arr, objPts);
	//cvTranspose(&imgPts_arr, imgPts);
	cvTranspose(&cvMat(3, correspondenceCount * imageCount, CV_64FC1, (void*)worldPts), objPts);
	cvTranspose(&cvMat(2, correspondenceCount * imageCount, CV_64FC1, (void*)imagePts), imgPts);
/*
	// method #2
	CvMat* objPts_arr = cvCreateMat(3, correspondenceCount * imageCount, CV_64FC1);
	CvMat* imgPts_arr = cvCreateMat(2, correspondenceCount * imageCount, CV_64FC1);
	cvSetData(objPts_arr, (void*)worldPts, sizeof(double) * correspondenceCount * imageCount);
	cvSetData(imgPts_arr, (void*)imagePts, sizeof(double) * correspondenceCount * imageCount);
	CvMat* objPts = cvCreateMat(correspondenceCount * imageCount, 3, CV_64FC1);
	CvMat* imgPts = cvCreateMat(correspondenceCount * imageCount, 2, CV_64FC1);
	cvTranspose(objPts_arr, objPts);
	cvTranspose(imgPts_arr, imgPts);
	cvReleaseMat(&objPts_arr);
	cvReleaseMat(&imgPts_arr);
*/
	//print_opencv_matrix(objPts);
	//print_opencv_matrix(imgPts);

	CvMat* ptCounts = cvCreateMat(imageCount, 1, CV_32SC1);
	CvMat* intrinsicParams = cvCreateMat(3, 3, CV_64FC1);
	CvMat* distortionCoeffs = cvCreateMat(4, 1, CV_64FC1);
	CvMat* rotation = cvCreateMat(imageCount, 3, CV_64FC1);
	CvMat* translation = cvCreateMat(imageCount, 3, CV_64FC1);

	cvSet2D(ptCounts, 0, 0, cvScalar(correspondenceCount));
	cvSet2D(ptCounts, 1, 0, cvScalar(correspondenceCount));

	int flags = 0;
	if (isPlanarCalibrationRigs)
	{
		// TODO [check] >>
		cvSetIdentity(intrinsicParams);
		//cvmSet(intrinsicParams, 0, 0, alpha_x);  // alpha_x
		//cvmSet(intrinsicParams, 0, 1, s);  // s
		//cvmSet(intrinsicParams, 0, 2, x0);  // x0
		//cvmSet(intrinsicParams, 1, 1, alpha_y);  // alpha_y
		//cvmSet(intrinsicParams, 1, 1, y0);  // y0

		flags = CV_CALIB_FIX_ASPECT_RATIO;
	}
	else
	{
		// TODO [check] >>
		cvSetIdentity(intrinsicParams);
		cvmSet(intrinsicParams, 0, 0, 1.0);  // alpha_x
		cvmSet(intrinsicParams, 0, 1, 0.0);  // s
		cvmSet(intrinsicParams, 0, 2, imgWidth * 0.5);  // x0
		cvmSet(intrinsicParams, 1, 1, 1.0);  // alpha_y
		cvmSet(intrinsicParams, 1, 1, imgHeight * 0.5);  // y0

		flags = CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_FIX_ASPECT_RATIO;
	}

	//
	cvCalibrateCamera2(objPts, imgPts, ptCounts, cvSize(imgWidth, imgHeight), intrinsicParams, distortionCoeffs, rotation, translation, flags);

	std::cout << ">>> intrinsic params =" << std::endl;
	print_opencv_matrix(intrinsicParams);
	std::cout << ">>> distortion coeffs =" << std::endl;
	print_opencv_matrix(distortionCoeffs);
	std::cout << ">>> rotation vectors =" << std::endl;
	print_opencv_matrix(rotation);
	std::cout << ">>> translation vectors =" << std::endl;
	print_opencv_matrix(translation);

	cvReleaseMat(&objPts);
	cvReleaseMat(&imgPts);
	cvReleaseMat(&ptCounts);
	cvReleaseMat(&intrinsicParams);
	cvReleaseMat(&distortionCoeffs);
	cvReleaseMat(&rotation);
	cvReleaseMat(&translation);
}

void extrinsic_camera_params()
{
	const int correspondenceCount = 12;

	{
/*
		const double worldPts[] = {
			-6375.79, -7292.85, -8963.5, -9109.77, -8339.89, -6130.0, -6528.26, -22746.9, -8442.19, -7879.51, -6835.6, -7889.53,
			-14763.6, -16615.2, -19982.2, -20267.9, -18716.2, -14246.9, -15043.4, -50051.5, -19103.9, -18188.7, -15952.3, -18152.4,
			-121.166, -136.389, -164.888, -166.048, -152.894, -115.55, -122.866, -414.82, -157.763, -149.995, -131.236, -150.07,
		};
*/
		// normalized 8-point algorithm
		const double worldPts[] = {
			-2661.08, 4239.22, 1394.28, 1554.19, 2078.23, -1724.04, -4194.6, 565.549, 1319.88, 1159.02, 5971.86, 1161.29,
			-2484.6, 2833.56, 453.964, 871.437, 1461.99, -2552.14, -4379.67, 62.372, 442.012, 650.432, 5959.46, 414.166,
			-4.97828, 5.37613, 1.34403, 1.27898, 1.83671, -2.43475, -5.94024, 0.492512, 1.6971, 2.37888, 16.490, 2.35034,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		};
		const double imagePts[] = {
			535, 789, 1037, 1215, 1131, 709, 707, 1150, 778, 487, 365, 495,
			500, 528, 337, 681, 795, 1050, 739, 130, 261, 273, 367, 178,
		};
		const double calibrationMat[] = {
			//1899.5976324404778, 0.0, 901.02286744183607, 0.0, 1899.5976324404778, 869.37281974232121, 0.0, 0.0, 1.1976796248217405
			1885.6097240797815, 0.0, 914.71724853916419, 0.0, 1885.6097240797815, 880.01640227244650, 0.0, 0.0, 1.2054610012686870
		};

		CvMat* objPts = cvCreateMat(correspondenceCount, 3, CV_64FC1);
		CvMat* imgPts = cvCreateMat(correspondenceCount, 2, CV_64FC1);
		const CvMat intrinsicParams = cvMat(3, 3, CV_64FC1, (void*)calibrationMat);
		cvTranspose(&cvMat(3, correspondenceCount, CV_64FC1, (void*)worldPts), objPts);
		cvTranspose(&cvMat(2, correspondenceCount, CV_64FC1, (void*)imagePts), imgPts);

		//print_opencv_matrix(objPts);
		//print_opencv_matrix(imgPts);
		//print_opencv_matrix(&intrinsicParams);

		CvMat* distortionCoeffs = cvCreateMat(4, 1, CV_64FC1);
		cvSetZero(distortionCoeffs);
		CvMat* rotation = cvCreateMat(1, 3, CV_64FC1);
		CvMat* translation = cvCreateMat(1, 3, CV_64FC1);

		//
		cvFindExtrinsicCameraParams2(objPts, imgPts, &intrinsicParams, distortionCoeffs, rotation, translation);

		std::cout << ">>> rotation vector =" << std::endl;
		print_opencv_matrix(rotation);
		std::cout << ">>> translation vector =" << std::endl;
		print_opencv_matrix(translation);

		//
		CvMat* essentialMat = cvCreateMat(3, 3, CV_64FC1);

		essential_matrix(rotation, translation, essentialMat);

		std::cout << ">>> essential matrix =" << std::endl;
		print_opencv_matrix(essentialMat);

		cvReleaseMat(&essentialMat);
		cvReleaseMat(&objPts);
		cvReleaseMat(&imgPts);
		cvReleaseMat(&distortionCoeffs);
		cvReleaseMat(&rotation);
		cvReleaseMat(&translation);
	}

	{
/*
		const double worldPts[] = {
			-6375.79, -7292.85, -8963.5, -9109.77, -8339.89, -6130.0, -6528.26, -22746.9, -8442.19, -7879.51, -6835.6, -7889.53,
			-14763.6, -16615.2, -19982.2, -20267.9, -18716.2, -14246.9, -15043.4, -50051.5, -19103.9, -18188.7, -15952.3, -18152.4,
			-121.166, -136.389, -164.888, -166.048, -152.894, -115.55, -122.866, -414.82, -157.763, -149.995, -131.236, -150.07,
		};
*/
		// normalized 8-point algorithm
		const double worldPts[] = {
			-2661.08, 4239.22, 1394.28, 1554.19, 2078.23, -1724.04, -4194.6, 565.549, 1319.88, 1159.02, 5971.86, 1161.29,
			-2484.6, 2833.56, 453.964, 871.437, 1461.99, -2552.14, -4379.67, 62.372, 442.012, 650.432, 5959.46, 414.166,
			-4.97828, 5.37613, 1.34403, 1.27898, 1.83671, -2.43475, -5.94024, 0.492512, 1.6971, 2.37888, 16.490, 2.35034,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		};
		const double imagePts[] = {
			26, 232, 457, 551, 468, 78, 113, 1143, 367, 239, 105, 257,
			460, 511, 358, 686, 783, 982, 702, 57, 233, 175, 258, 76,
		};
		const double calibrationMat[] = {
			//1807.5455582776883, 0.0, 1527.8862141931654, 0.0, 1807.5455582776883, 1464.7024928329672, 0.0, 0.0, 1.5398479622111263
			1807.7295336058307, 0.0, 1155.5672350222146, 0.0, 1807.7295336058307, 1068.6019537105219, 0.0, 0.0, 1.3259176768564587
		};

		CvMat* objPts = cvCreateMat(correspondenceCount, 3, CV_64FC1);
		CvMat* imgPts = cvCreateMat(correspondenceCount, 2, CV_64FC1);
		const CvMat intrinsicParams = cvMat(3, 3, CV_64FC1, (void*)calibrationMat);
		cvTranspose(&cvMat(3, correspondenceCount, CV_64FC1, (void*)worldPts), objPts);
		cvTranspose(&cvMat(2, correspondenceCount, CV_64FC1, (void*)imagePts), imgPts);

		//print_opencv_matrix(objPts);
		//print_opencv_matrix(imgPts);
		//print_opencv_matrix(&intrinsicParams);

		CvMat* distortionCoeffs = cvCreateMat(4, 1, CV_64FC1);
		cvSetZero(distortionCoeffs);
		CvMat* rotation = cvCreateMat(1, 3, CV_64FC1);
		CvMat* translation = cvCreateMat(1, 3, CV_64FC1);

		//
		cvFindExtrinsicCameraParams2(objPts, imgPts, &intrinsicParams, distortionCoeffs, rotation, translation);

		std::cout << ">>> rotation vector =" << std::endl;
		print_opencv_matrix(rotation);
		std::cout << ">>> translation vector =" << std::endl;
		print_opencv_matrix(translation);

		//
		CvMat* essentialMat = cvCreateMat(3, 3, CV_64FC1);

		essential_matrix(rotation, translation, essentialMat);

		std::cout << ">>> essential matrix =" << std::endl;
		print_opencv_matrix(essentialMat);

		cvReleaseMat(&essentialMat);
		cvReleaseMat(&objPts);
		cvReleaseMat(&imgPts);
		cvReleaseMat(&distortionCoeffs);
		cvReleaseMat(&rotation);
		cvReleaseMat(&translation);
	}
}

void fundamental_matrix()
{
	const int correspondenceCount = 12;

	const double imagePts1[] = {
		535, 789, 1037, 1215, 1131, 709, 707, 1150, 778, 487, 365, 495,
		500, 528, 337, 681, 795, 1050, 739, 130, 261, 273, 367, 178,
	};
	const double imagePts2[] = {
		26, 232, 457, 551, 468, 78, 113, 1143, 367, 239, 105, 257,
		460, 511, 358, 686, 783, 982, 702, 57, 233, 175, 258, 76,
	};

	CvMat* fundamentalMat = cvCreateMat(3, 3, CV_64FC1);

	//
	//const int method = CV_FM_7POINT;
	//const int method = CV_FM_8POINT;
	const int method = CV_FM_RANSAC;
	//const int method = CV_FM_LMEDS;
	cvFindFundamentalMat(&cvMat(2, correspondenceCount, CV_64FC1, (void*)imagePts1), &cvMat(2, correspondenceCount, CV_64FC1, (void*)imagePts2), fundamentalMat, method, 1.0, 0.9, NULL);
/*
	CvMat* imgPts1 = cvCreateMat(correspondenceCount, 2, CV_64FC1);
	cvTranspose(&cvMat(2, correspondenceCount, CV_64FC1, (void*)imagePts1), imgPts1);
	CvMat* imgPts2 = cvCreateMat(correspondenceCount, 2, CV_64FC1);
	cvTranspose(&cvMat(2, correspondenceCount, CV_64FC1, (void*)imagePts2), imgPts2);

	cvFindFundamentalMat(imgPts1, imgPts2, fundamentalMat, method, 1.0, 0.99, NULL);

	cvReleaseMat(&imgPts1);
	cvReleaseMat(&imgPts2);
*/
	std::cout << ">>> fundamental matrix =" << std::endl;
	print_opencv_matrix(fundamentalMat);

	// camera matrix
	CvMat* P = cvCreateMat(3, 4, CV_64FC1);
	CvMat* Pp= cvCreateMat(3, 4, CV_64FC1);

	camera_matrix(fundamentalMat, P, Pp);

	std::cout << ">>> camera matrix, P =" << std::endl;
	print_opencv_matrix(P);
	std::cout << ">>> camera matrix, P' =" << std::endl;
	print_opencv_matrix(Pp);

	{
/*
		// from fundamental matrix by OpenCV (RANSAC)
		const double worldPts[] = {
			-237.284, -286.76, -211.318, -507.963, 294.396, 368.306, -564.07, -2642.36, -262.828, -452.599, -323.59, -383.079, 
			-221.346, -191.868, -69.296, -284.743, 206.95, 545.618, -589.86, -300.123, -88.1931, -253.845, -319.251, -137.862, 
			-0.443655, -0.363451, -0.203656, -0.418073, 0.260298, 0.519506, -0.797832, -2.29736, -0.337817, -0.929263, -0.892615, -0.773805, 
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		};
*/
/*
		// normalized 8-point algorithm
		const double worldPts[] = {
			-2661.08, 4239.22, 1394.28, 1554.19, 2078.23, -1724.04, -4194.6, 565.549, 1319.88, 1159.02, 5971.86, 1161.29,
			-2484.6, 2833.56, 453.964, 871.437, 1461.99, -2552.14, -4379.67, 62.372, 442.012, 650.432, 5959.46, 414.166,
			-4.97828, 5.37613, 1.34403, 1.27898, 1.83671, -2.43475, -5.94024, 0.492512, 1.6971, 2.37888, 16.490, 2.35034,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		};
*/
		// gold standard algorithm
		const double worldPts[] = {
			-1757.2, -1899.02, -2040.06, -1740.36, -1681.35, -1559.64, -1718.06, 7617.01, -2485.62, -3107.88, -2352.11, -2700.05, 
			-1738.51, -1290.53, -593.144, -914.247, -1137.14, -2276.97, -1821.03, -88.2632, -876.719, -1994.88, -2410.09, -1278.06, 
			-2.89832, -2.37988, -2.03237, -1.45603, -1.50094, -2.20518, -2.40227, 7.55785, -3.12209, -5.159, -4.57036, -4.31562, 
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 
		};

		CvMat X = cvMat(4, correspondenceCount, CV_64FC1, (void*)worldPts);  // caution !!!: row-major matrix
		CvMat* x = cvCreateMat(3, correspondenceCount, CV_64FC1);
		CvMat* xp= cvCreateMat(3, correspondenceCount, CV_64FC1);
		cvMatMul(P, &X, x);
		cvMatMul(Pp, &X, xp);
		for (int i = 0; i < correspondenceCount; ++i)
			for (int j = 0; j < 3; ++j)
			{
				cvmSet(x, j, i, cvmGet(x, j, i) / cvmGet(x, 2, i));
				cvmSet(xp, j, i, cvmGet(xp, j, i) / cvmGet(xp, 2, i));
			}
		std::cout << ">>> image point, x =" << std::endl;
		print_opencv_matrix(x);
		std::cout << ">>> image point, x' =" << std::endl;
		print_opencv_matrix(xp);

		cvReleaseMat(&x);
		cvReleaseMat(&xp);
	}

	{
		CvMat* F2 = cvCreateMat(3, 3, CV_64FC1);
		fundamental_matrix(P, Pp, F2);
		std::cout << ">>> recalculated fundamental matrix, F =" << std::endl;
		print_opencv_matrix(F2);
		cvReleaseMat(&F2);
	}

	cvReleaseMat(&P);
	cvReleaseMat(&Pp);

	cvReleaseMat(&fundamentalMat);
}

void camera_matrix(const CvMat* fundamental_matrix, CvMat* P, CvMat* P_prime)
{
	const int dim = 3;
	CvMat* FT = cvCreateMat(dim, dim, CV_64FC1);
	CvMat* U = cvCreateMat(dim, dim, CV_64FC1);
	CvMat* V = cvCreateMat(dim, dim, CV_64FC1);
	CvMat* W = cvCreateMat(dim, dim, CV_64FC1);

	cvTranspose(fundamental_matrix, FT);
	cvSVD(FT, W, U, V, 0);  // 0 or CV_SVD_MODIFY_A or CV_SVD_U_T or CV_SVD_V_T

	CvMat* Ep = cvCreateMat(dim, 1, CV_64FC1);
	cvGetCol(V, Ep, dim - 1);

	cvReleaseMat(&FT);
	cvReleaseMat(&U);
	cvReleaseMat(&V);
	cvReleaseMat(&W);

	CvMat* skewEp = cvCreateMat(dim, dim, CV_64FC1);
	cvSetZero(skewEp);
	cvmSet(skewEp, 0, 1, -cvmGet(Ep, 2, 0));
	cvmSet(skewEp, 0, 2, cvmGet(Ep, 1, 0));
	cvmSet(skewEp, 1, 0, cvmGet(Ep, 2, 0));
	cvmSet(skewEp, 1, 2, -cvmGet(Ep, 0, 0));
	cvmSet(skewEp, 2, 0, -cvmGet(Ep, 1, 0));
	cvmSet(skewEp, 2, 1, cvmGet(Ep, 0, 0));

	cvSetIdentity(P);
	CvMat* EpF = cvCreateMat(dim, dim, CV_64FC1);
	cvGetSubRect(P_prime, EpF, cvRect(0, 0, dim, dim));
	cvMatMul(skewEp, fundamental_matrix, EpF);
	for (int i = 0; i < dim; ++i)
		cvmSet(P_prime, i, dim, cvmGet(Ep, i, 0));

	cvReleaseMat(&Ep);
	cvReleaseMat(&skewEp);
	cvReleaseMat(&EpF);
}

void fundamental_matrix(const CvMat* P, const CvMat* P_prime, CvMat* fundamental_matrix)
{
	const int dim = 3;
	CvMat* U = cvCreateMat(dim, dim, CV_64FC1);
	CvMat* V = cvCreateMat(4, 4, CV_64FC1);
	CvMat* W = cvCreateMat(dim, 4, CV_64FC1);

	cvSVD((CvArr*)P, W, U, V, 0);  // 0 or CV_SVD_MODIFY_A or CV_SVD_U_T or CV_SVD_V_T
	CvMat* C = cvCreateMat(4, 1, CV_64FC1);
	cvGetCol(V, C, dim);

	cvReleaseMat(&U);
	cvReleaseMat(&V);
	cvReleaseMat(&W);

	CvMat* Ep = cvCreateMat(dim, 1, CV_64FC1);
	cvMatMul(P_prime, C, Ep);

	cvReleaseMat(&C);

	//
	CvMat* skewEp = cvCreateMat(dim, dim, CV_64FC1);
	cvSetZero(skewEp);
	cvmSet(skewEp, 0, 1, -cvmGet(Ep, 2, 0));
	cvmSet(skewEp, 0, 2, cvmGet(Ep, 1, 0));
	cvmSet(skewEp, 1, 0, cvmGet(Ep, 2, 0));
	cvmSet(skewEp, 1, 2, -cvmGet(Ep, 0, 0));
	cvmSet(skewEp, 2, 0, -cvmGet(Ep, 1, 0));
	cvmSet(skewEp, 2, 1, cvmGet(Ep, 0, 0));

	CvMat* PT = cvCreateMat(4, dim, CV_64FC1);
	CvMat* P_PT = cvCreateMat(dim, dim, CV_64FC1);
	CvMat* P_PT_inv = cvCreateMat(dim, dim, CV_64FC1);
	CvMat* P_pinv = cvCreateMat(4, dim, CV_64FC1);
	cvTranspose(P, PT);
	cvMatMul(P, PT, P_PT);
	cvInvert(P_PT, P_PT_inv, CV_LU);
	cvMatMul(PT, P_PT_inv, P_pinv);

	CvMat* Ep_Pp = cvCreateMat(dim, 4, CV_64FC1);
	cvMatMul(skewEp, P_prime, Ep_Pp);
	cvMatMul(Ep_Pp, P_pinv, fundamental_matrix);

	cvReleaseMat(&PT);
	cvReleaseMat(&P_PT);
	cvReleaseMat(&P_PT_inv);
	cvReleaseMat(&P_pinv);
	cvReleaseMat(&Ep_Pp);

	cvReleaseMat(&Ep);
	cvReleaseMat(&skewEp);
}

void gold_standard_algorithm()
{
}

void triangulation()
{
}
