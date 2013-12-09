//include "stdafx.h"
#if defined(__cplusplus)
extern "C" {
#endif
#include "../klt_lib/pnmio.h"
#include "../klt_lib/klt.h"
#if defined(__cplusplus)
}  // extern "C"
#endif
#include <sstream>
#include <iostream>


namespace {
namespace local {

// [ref] ${KLT_HOME}/example1.c.
void example1()
{
	const int nFeatures = 100;

	KLT_TrackingContext tc = KLTCreateTrackingContext();
	KLTPrintTrackingContext(tc);
	KLT_FeatureList fl = KLTCreateFeatureList(nFeatures);

	int ncols, nrows;
	unsigned char *img1 = pgmReadFile("./data/motion_analysis/klt/img0.pgm", NULL, &ncols, &nrows);
	unsigned char *img2 = pgmReadFile("./data/motion_analysis/klt/img1.pgm", NULL, &ncols, &nrows);

	KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

	std::cout << "\nIn first image:" << std::endl;
	for (int i = 0 ; i < fl->nFeatures ; ++i)
		std::cout << "Feature #" << i << ":  (" << fl->feature[i]->x << ',' << fl->feature[i]->y << ") with value of " << fl->feature[i]->val << std::endl;

	KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "./data/motion_analysis/klt/feat1.ppm");
	KLTWriteFeatureList(fl, "./data/motion_analysis/klt/feat1.txt", "%3d");

	KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

	std::cout << "\nIn second image:" << std::endl;
	for (int i = 0 ; i < fl->nFeatures ; ++i)
		std::cout << "Feature #" << i << ":  (" << fl->feature[i]->x << ',' << fl->feature[i]->y << ") with value of " << fl->feature[i]->val << std::endl;

	KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "./data/motion_analysis/klt/feat2.ppm");
	KLTWriteFeatureList(fl, "./data/motion_analysis/klt/feat2.fl", NULL);  // binary file.
	KLTWriteFeatureList(fl, "./data/motion_analysis/klt/feat2.txt", "%5.1f");  // text file.

	// Clean-up.
	KLTFreeFeatureList(fl);
	KLTFreeTrackingContext(tc);

	free(img1);
	free(img2);
}

// [ref] ${KLT_HOME}/example2.c.
void example2()
{
	const int nFeatures = 100;

	KLT_TrackingContext tc = KLTCreateTrackingContext();
	KLT_FeatureList fl = KLTCreateFeatureList(nFeatures);

	int ncols, nrows;
	unsigned char *img1 = pgmReadFile("./data/motion_analysis/klt/img0.pgm", NULL, &ncols, &nrows);
	unsigned char *img2 = pgmReadFile("./data/motion_analysis/klt/img1.pgm", NULL, &ncols, &nrows);

	KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

	KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "./data/motion_analysis/klt/feat1.ppm");
	KLTWriteFeatureList(fl, "./data/motion_analysis/klt/feat1.txt", "%3d");

	KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
	KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);

	KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "./data/motion_analysis/klt/feat2.ppm");
	KLTWriteFeatureList(fl, "./data/motion_analysis/klt/feat2.txt", "%3d");

	// Clean-up.
	KLTFreeFeatureList(fl);
	KLTFreeTrackingContext(tc);

	free(img1);
	free(img2);
}

// [ref] ${KLT_HOME}/example3.c.
void example3()
{
	const int nFeatures = 150, nFrames = 10;

	KLT_TrackingContext tc = KLTCreateTrackingContext();
	KLT_FeatureList fl = KLTCreateFeatureList(nFeatures);
	KLT_FeatureTable ft = KLTCreateFeatureTable(nFrames, nFeatures);
	tc->sequentialMode = TRUE;
	tc->writeInternalImages = FALSE;
	tc->affineConsistencyCheck = -1;  // set this to 2 to turn on affine consistency check.

	int ncols, nrows;
	unsigned char *img1 = pgmReadFile("./data/motion_analysis/klt/img0.pgm", NULL, &ncols, &nrows);
	unsigned char *img2 = (unsigned char *)malloc(ncols * nrows * sizeof(unsigned char));

	KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
	KLTStoreFeatureList(fl, ft, 0);
	KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "./data/motion_analysis/klt/feat0.ppm");

	for (int i = 1 ; i < nFrames ; ++i)
	{
		{
			std::ostringstream strm;
			strm << "./data/motion_analysis/klt/img" << i << ".pgm";
			pgmReadFile((char *)strm.str().c_str(), img2, &ncols, &nrows);
		}
		KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
#ifdef REPLACE
		KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
		KLTStoreFeatureList(fl, ft, i);
		{
			std::ostringstream strm;
			strm << "./data/motion_analysis/klt/feat" << i << ".ppm";
			KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, (char *)strm.str().c_str());
		}
	}
	KLTWriteFeatureTable(ft, "./data/motion_analysis/klt/features.txt", "%5.1f");
	KLTWriteFeatureTable(ft, "./data/motion_analysis/klt/features.ft", NULL);

	// Clean-up.
	KLTFreeFeatureTable(ft);
	KLTFreeFeatureList(fl);
	KLTFreeTrackingContext(tc);

	free(img1);
	free(img2);
}

// [ref] ${KLT_HOME}/example4.c.
void example4()
{
	KLT_FeatureTable ft = KLTReadFeatureTable(NULL, "./data/motion_analysis/klt/features.txt");
	KLT_FeatureList fl = KLTCreateFeatureList(ft->nFeatures);
	KLTExtractFeatureList(fl, ft, 1);
	KLTWriteFeatureList(fl, "./data/motion_analysis/klt/feat1.txt", "%3d");
	KLTReadFeatureList(fl, "./data/motion_analysis/klt/feat1.txt");
	KLTStoreFeatureList(fl, ft, 2);
	KLTWriteFeatureTable(ft, "./data/motion_analysis/klt/ft2.txt", "%3d");

	KLT_FeatureHistory fh = KLTCreateFeatureHistory(ft->nFrames);
	KLTExtractFeatureHistory(fh, ft, 5);

	std::cout << "The feature history of feature number 5:" << std::endl << std::endl;
	for (int i = 0 ; i < fh->nFrames ; ++i)
		std::cout << i << ": (" << fh->feature[i]->x << ',' << fh->feature[i]->y << ") = " << fh->feature[i]->val << std::endl;

	KLTStoreFeatureHistory(fh, ft, 8);
	KLTWriteFeatureTable(ft, "./data/motion_analysis/klt/ft3.txt", "%6.1f");

	// Clean-up.
	KLTFreeFeatureHistory(fh);
	KLTFreeFeatureTable(ft);
	KLTFreeFeatureList(fl);
}

// [ref] ${KLT_HOME}/example5.c.
void example5()
{
	const int nFeatures = 100;

	KLT_TrackingContext tc = KLTCreateTrackingContext();
	tc->mindist = 20;
	tc->window_width  = 9;
	tc->window_height = 9;
	KLTChangeTCPyramid(tc, 15);
	KLTUpdateTCBorder(tc);
	KLT_FeatureList fl = KLTCreateFeatureList(nFeatures);

	int ncols, nrows;
	unsigned char *img1 = pgmReadFile("./data/motion_analysis/klt/img0.pgm", NULL, &ncols, &nrows);
	unsigned char *img2 = pgmReadFile("./data/motion_analysis/klt/img2.pgm", NULL, &ncols, &nrows);

	KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

	KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "./data/motion_analysis/klt/feat1b.ppm");

	KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

	KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "./data/motion_analysis/klt/feat2b.ppm");

	// Clean-up.
	KLTFreeFeatureList(fl);
	KLTFreeTrackingContext(tc);

	free(img1);
	free(img2);
}

}  // namespace local
}  // unnamed namespace

namespace my_klt {

}  // namespace my_klt

int klt_main(int argc, char *argv[])
{
	//local::example1();
	//local::example2();
	//local::example3();
	local::example4();  // example4 reads output from example3.
	local::example5();

	return 0;
}
