//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <map>
#include <string>
#include <iostream>
#include <ctime>


namespace {
namespace local {

void kmeans()
{
	const size_t MAX_CLUSTERS = 5;

	CvScalar color_tab[MAX_CLUSTERS];
	IplImage *img = cvCreateImage(cvSize(500, 500), 8, 3);
	CvRNG rng = cvRNG(-1);
	CvPoint ipt;

	color_tab[0] = CV_RGB(255, 0, 0);
	color_tab[1] = CV_RGB(0, 255, 0);
	color_tab[2] = CV_RGB(100, 100 ,255);
	color_tab[3] = CV_RGB(255, 0, 255);
	color_tab[4] = CV_RGB(255, 255, 0);

	cvNamedWindow("clusters", 1);

	for (;;)
	{
		char key;
		const unsigned int sample_count = cvRandInt(&rng) % 1000 + 1;
		const unsigned int cluster_count = MIN(cvRandInt(&rng) % MAX_CLUSTERS + 1, sample_count);
		CvMat *points = cvCreateMat(sample_count, 1, CV_32FC2);
		CvMat *clusters = cvCreateMat(sample_count, 1, CV_32SC1);
		CvMat *cluster_centers = cvCreateMat(cluster_count, 1, CV_32FC2);

		// generate random sample from multigaussian distribution
		for (unsigned int k = 0; k < cluster_count; ++k)
		{
#if 0
			CvPoint center;
			CvMat point_chunk;
			center.x = cvRandInt(&rng) % img->width;
			center.y = cvRandInt(&rng) % img->height;
			cvGetRows(
				points, &point_chunk, k * sample_count / cluster_count,
				k == cluster_count - 1 ? sample_count : (k+1)*sample_count/cluster_count, 1
			);

			cvRandArr(
				&rng, &point_chunk, CV_RAND_NORMAL,
				cvScalar(center.x, center.y, 0, 0),
				cvScalar(img->width * 0.1, img->height * 0.1, 0, 0)
			);
#else
			// directional data
			CvPoint center;
			CvMat point_chunk;
			const float angle = (cvRandInt(&rng) % 36) * 10.0f;
			const float radius = MIN(img->width, img->height) / 4.0f;
			center.x = img->width / 2 + int(radius * std::cos(angle));
			center.y = img->height / 2 + int(radius * std::sin(angle));
			//center.x = int(radius * std::cos(angle));
			//center.y = int(radius * std::sin(angle));

			//cvGetRows(
			//	points, &point_chunk, k * sample_count / cluster_count,
			//	k == cluster_count - 1 ? sample_count : (k+1)*sample_count/cluster_count, 1
			//);

			//cvRandArr(
			//	&rng, &point_chunk, CV_RAND_NORMAL,
			//	cvScalar(center.x, center.y, 0, 0),
			//	cvScalar(img->width * 0.01, img->height * 0.01, 0, 0)
			//);

			const size_t start_idx = k * sample_count / cluster_count;
			const size_t end_idx = k == cluster_count - 1 ? sample_count : (k+1)*sample_count/cluster_count;

			for (size_t i = start_idx; i < end_idx; ++i)
			{
				const float delta_angle = (cvRandInt(&rng) % 101) / 100.0f;
				cvSet1D(points, i, cvScalar(center.x + (float)std::cos(angle + delta_angle), center.y + (float)std::sin(angle + delta_angle), 0, 0));
			}
#endif
		}

		// shuffle samples
		for (unsigned int i = 0; i < sample_count / 2; ++i)
		{
			CvPoint2D32f *pt1 = (CvPoint2D32f*)points->data.fl + cvRandInt(&rng) % sample_count;
			CvPoint2D32f *pt2 = (CvPoint2D32f*)points->data.fl + cvRandInt(&rng) % sample_count;
			CvPoint2D32f temp;
			CV_SWAP(*pt1, *pt2, temp);
		}

		const int iteration_count = cvKMeans2(
			points, cluster_count, clusters,
			cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 0.001),
			5, 0, 0, cluster_centers, 0
		);
		std::cout << "# of clusters: " << cluster_count <<", # of iterations: " << iteration_count << std::endl;

#if 0
		// Oops !!!
		// cluster centers can not be used. these are not meaningful data
		for (unsigned int k = 0; k < cluster_count; ++k)
		{
			const CvScalar cntr = cvGet1D(cluster_centers, k);
			std::cout << "(" << cntr.val[0] << "," << cntr.val[1] << "," << cntr.val[2] << "," << cntr.val[3] << "), ";
		}
		std::cout << std::endl;
#else
		std::vector<CvScalar> centers(cluster_count);
		std::vector<size_t> cluster_item_counts(cluster_count, 0);
		for (unsigned int k = 0; k < cluster_count; ++k)
			centers[k] = cvScalar(0.0, 0.0, 0.0, 0.0);

		for (unsigned int i = 0; i < sample_count; ++i)
		{
			const int idx = CV_MAT_ELEM(*clusters, int, i, 0);
			const CvScalar pt = cvGet1D(points, i);
			centers[idx].val[0] += pt.val[0];
			centers[idx].val[1] += pt.val[1];
			centers[idx].val[2] += pt.val[2];
			centers[idx].val[3] += pt.val[3];
			++cluster_item_counts[idx];
		}

		for (unsigned int k = 0; k < cluster_count; ++k)
		{
			const size_t sz = cluster_item_counts[k];
			if (0 == sz)
			{
				centers[k].val[0] = 0.0f;
				centers[k].val[1] = 0.0f;
				centers[k].val[2] = 0.0f;
				centers[k].val[3] = 0.0f;
			}
			else
			{
				centers[k].val[0] /= float(sz);
				centers[k].val[1] /= float(sz);
				centers[k].val[2] /= float(sz);
				centers[k].val[3] /= float(sz);
			}
		}

		// calculate cluster's centers
		for (unsigned int k = 0; k < cluster_count; ++k)
		{
			//const CvScalar cntr = cvGet1D(cluster_centers, k);
			//std::cout << "(" << cntr.val[0] << "," << cntr.val[1] << "," << cntr.val[2] << "," << cntr.val[3] << "), ";
			std::cout << "(" << centers[k].val[0] << "," << centers[k].val[1] << "," << centers[k].val[2] << "," << centers[k].val[3] << "), ";
		}
		std::cout << std::endl;
#endif

		cvZero(img);

		for (unsigned int i = 0; i < sample_count; ++i)
		{
			int cluster_idx = clusters->data.i[i];
			ipt.x = (int)points->data.fl[i*2];
			ipt.y = (int)points->data.fl[i*2+1];
			cvCircle(img, ipt, 2, color_tab[cluster_idx], CV_FILLED, CV_AA, 0);
		}

		cvReleaseMat(&points);
		cvReleaseMat(&clusters);
		cvReleaseMat(&cluster_centers);

		cvShowImage("clusters", img);

		key = (char)cvWaitKey(0);
		if (key == 27 || key == 'q' || key == 'Q')  // 'ESC'
			break;
	}

	cvDestroyWindow("clusters");
	cvReleaseImage(&img);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void clustering()
{
	local::kmeans();
}

}  // namespace my_opencv
