//#include "stdafx.h"
#include <opensift/sift.h>
#include <opensift/imgfeatures.h>
#include <opensift/kdtree.h>
#include <opensift/utils.h>
#include <opensift/xform.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <iostream>


namespace {
namespace local {

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

}  // namespace local
}  // unnamed namespace

namespace my_opensift {

// [ref] ${OPENSIFT_HOME}/src/match.c
void match_feature()
{
#if 1
    const std::string img1_file("./data/feature_analysis/sift/beaver.png");
    const std::string img2_file("./data/feature_analysis/sift/beaver_xform.png");
#elif 0
    const std::string img1_file("./data/feature_analysis/sift/marker_pen_3.bmp");
    const std::string img2_file("./data/feature_analysis/sift/marker_pen_test_image.bmp");
#elif 0
    const std::string img1_file("./data/feature_analysis/sift/melon_target.png");
    const std::string img2_file("./data/feature_analysis/sift/melon_3.png");
#endif
    //const std::string img_output_file("./data/feature_analysis/sift/marker_pen_sift_match_result_3.bmp");

	IplImage *img1 = cvLoadImage(img1_file.c_str(), 1);
	if (!img1)
		fatal_error((char *)"unable to load image from %s", (char *)img1_file.c_str());
	IplImage *img2 = cvLoadImage(img2_file.c_str(), 1);
	if (!img2)
		fatal_error((char *)"unable to load image from %s", (char *)img2_file.c_str());
	IplImage *stacked = stack_imgs(img1, img2);

    //
	struct feature *feat1, *feat2, *feat;

	std::cout << "finding features in " << img1_file << "..." << std::endl;
	const int n1 = sift_features(img1, &feat1);

	std::cout << "finding features in " << img2_file << "..." << std::endl;
	const int n2 = sift_features(img2, &feat2);

	struct kd_node *kd_root = kdtree_build(feat2, n2);
	struct feature **nbrs;
	double d0, d1;
	CvPoint pt1, pt2;
	int k, i, m = 0;
	for (i = 0; i < n1; ++i)
	{
		feat = feat1 + i;
		k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);
		if (2 == k)
		{
			d0 = descr_dist_sq(feat, nbrs[0]);
			d1 = descr_dist_sq(feat, nbrs[1]);
			if (d0 < d1 * NN_SQ_DIST_RATIO_THR)
			{
				pt1 = cvPoint(cvRound(feat->x), cvRound(feat->y));
				pt2 = cvPoint(cvRound(nbrs[0]->x), cvRound(nbrs[0]->y));
				pt2.y += img1->height;
				cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);
				++m;
				feat1[i].fwd_match = nbrs[0];
			}
		}

		free(nbrs);
	}

	//cvSaveImage(img_output_file.c_str(), stacked);

	std::cout << "found " << m << " total matches" << std::endl;
	cvNamedWindow("Matches", 1);
	cvShowImage("Matches", stacked);
	cvWaitKey(0);

/*
	UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS

	Note that this line above:

	feat1[i].fwd_match = nbrs[0];

	is important for the RANSAC function to work.
*/
	{
		CvMat *H = ransac_xform(feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01, homog_xfer_err, 3.0, NULL, NULL);
		if (H)
		{
			IplImage *xformed = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);

			cvWarpPerspective(img1, xformed, H, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

			cvNamedWindow("Xformed", 1);
			cvShowImage("Xformed", xformed);
			cvWaitKey(0);

			cvReleaseImage(&xformed);
			cvReleaseMat(&H);
		}
	}

	cvReleaseImage(&stacked);
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	kdtree_release(kd_root);
	free(feat1);
	free(feat2);
}

}  // namespace my_opensift
