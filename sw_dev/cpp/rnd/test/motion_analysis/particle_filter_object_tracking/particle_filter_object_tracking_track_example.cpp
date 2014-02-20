#include "../particle_filter_object_tracking_lib/defs.h"
#include "../particle_filter_object_tracking_lib/utils.h"
#include "../particle_filter_object_tracking_lib/particles.h"
#include "../particle_filter_object_tracking_lib/observation.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdio>
#include <ctime>


namespace {
namespace local {

/* maximum number of objects to be tracked */
#define MAX_OBJECTS 1

typedef struct params
{
	CvPoint loc1[MAX_OBJECTS];
	CvPoint loc2[MAX_OBJECTS];
	IplImage *objects[MAX_OBJECTS];
	char *win_name;
	IplImage *orig_img;
	IplImage *cur_img;
	int n;
} params;

/*
Mouse callback function that allows user to specify the initial object
regions.  Parameters are as specified in OpenCV documentation.
*/
void mouse(int event, int x, int y, int flags, void *param)
{
	params *p = (params *)param;
	CvPoint *loc;
	int n;
	IplImage *tmp;
	static int pressed = FALSE;

	// on left button press, remember first corner of rectangle around object
	if (CV_EVENT_LBUTTONDOWN == event)
	{
		n = p->n;
		if (MAX_OBJECTS == n)
			return;
		loc = p->loc1;
		loc[n].x = x;
		loc[n].y = y;
		pressed = TRUE;
	}

	// on left button up, finalize the rectangle and draw it in black
	else if (CV_EVENT_LBUTTONUP == event)
	{
		n = p->n;
		if (MAX_OBJECTS == n)
			return;
		loc = p->loc2;
		loc[n].x = x;
		loc[n].y = y;
		cvReleaseImage(&(p->cur_img));
		p->cur_img = NULL;
		cvRectangle(p->orig_img, p->loc1[n], loc[n], CV_RGB(0, 0, 0), 1, 8, 0);
		cvShowImage(p->win_name, p->orig_img);
		pressed = FALSE;
		p->n++;
	}

	// on mouse move with left button down, draw rectangle as defined in white
	else if (CV_EVENT_MOUSEMOVE == event && flags & CV_EVENT_FLAG_LBUTTON)
	{
		n = p->n;
		if (MAX_OBJECTS == n)
			return;
		tmp = (IplImage *)cvClone(p->orig_img);
		loc = p->loc1;
		cvRectangle(tmp, loc[n], cvPoint(x, y), CV_RGB(255, 255, 255), 1, 8, 0);
		cvShowImage(p->win_name, tmp);
		if (p->cur_img)
			cvReleaseImage(&(p->cur_img));
		p->cur_img = tmp;
	}
}

/*
Allows the user to interactively select object regions.

@param frame the frame of video in which objects are to be selected
@param regions a pointer to an array to be filled with rectangles
defining object regions

@return Returns the number of objects selected by the user
*/
int get_regions(IplImage *frame, CvRect **regions)
{
	char *win_name = "First frame";
	params p;
	CvRect *r;
	int i, x1, y1, x2, y2, w, h;

	// use mouse callback to allow user to define object regions
	p.win_name = win_name;
	p.orig_img = (IplImage *)cvClone( frame );
	p.cur_img = NULL;
	p.n = 0;
	cvNamedWindow(win_name, 1);
	cvShowImage(win_name, frame);
	cvSetMouseCallback(win_name, &mouse, &p);
	cvWaitKey(0);
	cvDestroyWindow(win_name);
	cvReleaseImage(&(p.orig_img));
	if (p.cur_img)
		cvReleaseImage(&(p.cur_img));

	// extract regions defined by user; store as an array of rectangles
	if (p.n == 0)
	{
		*regions = NULL;
		return 0;
	}
	r = (CvRect *)malloc(p.n * sizeof(CvRect));
	for (i = 0; i < p.n; ++i)
	{
		x1 = MIN(p.loc1[i].x, p.loc2[i].x);
		x2 = MAX(p.loc1[i].x, p.loc2[i].x);
		y1 = MIN(p.loc1[i].y, p.loc2[i].y);
		y2 = MAX(p.loc1[i].y, p.loc2[i].y);
		w = x2 - x1;
		h = y2 - y1;

		// ensure odd width and height
		w = (w % 2) ? w : w + 1;
		h = (h % 2) ? h : h + 1;
		r[i] = cvRect(x1, y1, w, h);
	}
	*regions = r;
	return p.n;
}

/*
Computes a reference histogram for each of the object regions defined by
the user

@param frame video frame in which to compute histograms; should have been
converted to hsv using bgr2hsv in observation.h
@param regions regions of \a frame over which histograms should be computed
@param n number of regions in \a regions
@param export if TRUE, object region images are exported

@return Returns an \a n element array of normalized histograms corresponding
to regions of \a frame specified in \a regions.
*/
histogram ** compute_ref_histos(IplImage *frame, CvRect *regions, int n)
{
	histogram **histos = (histogram **)malloc(n * sizeof(histogram *));
	IplImage *tmp;
	int i;

	// extract each region from frame and compute its histogram
	for (i = 0; i < n; ++i)
	{
		cvSetImageROI(frame, regions[i]);
		tmp = cvCreateImage(cvGetSize(frame), IPL_DEPTH_32F, 3);
		cvCopy(frame, tmp, NULL);
		cvResetImageROI(frame);
		histos[i] = calc_histogram(&tmp, 1);
		normalize_histogram(histos[i]);
		cvReleaseImage(&tmp);
	}

	return histos;
}

/*
Exports reference histograms to file

@param ref_histos array of reference histograms
@param n number of histograms

@return Returns 1 on success or 0 on failure
*/
int export_ref_histos(histogram **ref_histos, int n)
{
	char name[32];
	char num[3];
	FILE *file;
	int i;

	for (i = 0; i < n; ++i)
	{
#if defined(WIN32) || defined(_WIN32)
		_snprintf(num, 3, "%02d", i);
#else
		snprintf(num, 3, "%02d", i);
#endif
		strcpy(name, "hist_");
		strcat(name, num);
		strcat(name, ".dat");
		if (!export_histogram(ref_histos[i], name))
			return 0;
	}

	return 1;
}

/*
Exports a frame whose name and format are determined by export_base and
export_extension, defined above.

@param frame frame to be exported
@param i frame number
*/
int export_frame(IplImage *frame, int i)
{
	const std::string export_base("./data/motion_analysis/frame_");
	const std::string export_extension(".png");

	std::ostringstream num;
	num << std::setfill('0') << std::setw(4) << i;

	const std::string name(export_base + num.str() + export_extension);

	return cvSaveImage(name.c_str(), frame);
}

}  // namespace local
}  // unnamed namespace

namespace my_particle_filter_object_tracking {

// ${PARTICLE_FILTER_OBJECT_TRACKING_HOME}/src/track1.c.
void track_example()
{
	const std::string input_vided_file("./data/motion_analysis/hockey.avi");  // input video file name.
	//const std::string input_vided_file("./data/motion_analysis/soccer.avi");  // input video file name.

	const int num_particles = 1000;  // number of particles.
	const bool show_all = false;  // true to display all particles.
	const bool export_flag = false;  // true to exported tracking sequence.
	const int MAX_FRAMES = 2048;

	gsl_rng *rng;
	IplImage *frame, *hsv_frame, *frames[MAX_FRAMES];
	IplImage **hsv_ref_imgs;
	histogram **ref_histos;
	particle *particles, *new_particles;
	CvScalar color;
	CvRect *regions;
	int num_objects = 0;
	float s;
	int i, j, k, w, h, x, y;

	gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng, std::time(NULL));

	CvCapture *video = cvCaptureFromFile(input_vided_file.c_str());
	if (!video)
	{
		std::cerr << "input video file not found : " << input_vided_file << std::endl;
		return;
	}

	i = 0;
	while (frame = cvQueryFrame(video))
	{
		hsv_frame = bgr2hsv(frame);
		frames[i] = (IplImage *)cvClone(frame);

		// allow user to select object to be tracked in the first frame
		if (0 == i)
		{
			w = frame->width;
			h = frame->height;
			std::cout << "Select object region to track" << std::endl;
			while (0 == num_objects)
			{
				num_objects = local::get_regions(frame, &regions);
				if (0 == num_objects)
					std::cout << "Please select a object" << std::endl;
			}

			// compute reference histograms and distribute particles.
			ref_histos = local::compute_ref_histos(hsv_frame, regions, num_objects);
			if (export_flag)
				local::export_ref_histos(ref_histos, num_objects);
			particles = init_distribution(regions, ref_histos, num_objects, num_particles);
		}
		else
		{
			// perform prediction and measurement for each particle.
			for (j = 0; j < num_particles; ++j)
			{
				particles[j] = transition(particles[j], w, h, rng);
				s = particles[j].s;
				particles[j].w = likelihood(
					hsv_frame, cvRound(particles[j].y),
					cvRound(particles[j].x),
					cvRound(particles[j].width * s),
					cvRound(particles[j].height * s),
					particles[j].histo
				);
			}

			// normalize weights and resample a set of unweighted particles.
			normalize_weights(particles, num_particles);
			new_particles = resample(particles, num_particles);
			free(particles);
			particles = new_particles;
		}

		// display all particles if requested.
		qsort(particles, num_particles, sizeof(particle), &particle_cmp);
		if (show_all)
			for (j = num_particles - 1; j > 0; --j)
			{
				color = CV_RGB(0, 0, 255);
				display_particle(frames[i], particles[j], color);
			}

		// display most likely particle.
		color = CV_RGB(255, 0, 0);
		display_particle(frames[i], particles[0], color);
		cvNamedWindow("Video", 1);
		cvShowImage("Video", frames[i]);
		cvWaitKey(5);

		cvReleaseImage(&hsv_frame);
		++i;
	}
	cvReleaseCapture(&video);

	// export video frames, if export requested.
	if (export_flag)
		std::cout << "Exporting video frames... " << std::endl;

	for (j = 0; j < i; ++j)
	{
		if (export_flag)
		{
			progress(FALSE);
			local::export_frame(frames[j], j + 1);
		}
		cvReleaseImage(&frames[j]);
	}

	if (export_flag)
		progress(TRUE);
}

}  // namespace my_particle_filter_object_tracking
