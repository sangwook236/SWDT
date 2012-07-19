#include "bp_vision/image.h"
#include "bp_vision/misc.h"
#include "bp_vision/pnmfile.h"
#include "bp_vision/filter.h"
#include "bp_vision/imconv.h"
#include <boost/multi_array.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <cassert>


namespace {
namespace local {

// computation of data costs
image<float *> * compute_data_costs(image<uchar> *img, const float LAMBDA, const float DATA_K, const int LABEL_NUM)
{
	const int width = img->width();
	const int height = img->height();
	image<float *> *data = new image<float *>(width, height);

	boost::multi_array<std::vector<float>, 2> data(boost::extents[width][height]);

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
			for (int value = 0; value < LABEL_NUM; ++value)
			{
				float val = square(float(imRef(img, x, y) - value));
				imRef(data, x, y)[value] = LAMBDA * std::min(val, DATA_K);
			}

	return data;
}

// computation of data costs
image<float *> * compute_data_costs(image<uchar> *img1, image<uchar> *img2, const float LAMBDA, const float DATA_K, const int LABEL_NUM, const double SIGMA)
{
	const int width = img1->width();
	const int height = img1->height();
	image<float *> *data = new image<float *>(width, height);

	image<float> *sm1, *sm2;
	if (SIGMA >= 0.1)
	{
		sm1 = smooth(img1, SIGMA);
		sm2 = smooth(img2, SIGMA);
	}
	else
	{
		sm1 = imageUCHARtoFLOAT(img1);
		sm2 = imageUCHARtoFLOAT(img2);
	} 

	for (int y = 0; y < height; ++y)
		for (int x = LABEL_NUM - 1; x < width; ++x)
			for (int value = 0; value < LABEL_NUM; ++value)
			{
				float val = std::abs(imRef(sm1, x, y) - imRef(sm2, x-value, y));	
				imRef(data, x, y)[value] = LAMBDA * std::min(val, DATA_K);
			}

	delete sm1;
	delete sm2;

	return data;
}

// compute message
void compute_message(float *s1, float *s2, float *s3, float *s4, float *dst, const float DISC_K, const int LABEL_NUM)
{
	// aggregate and find min
	float minimum = std::numeric_limits<float>::max();
	for (int value = 0; value < LABEL_NUM; ++value)
	{
		dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
		if (dst[value] < minimum)
			minimum = dst[value];
	}

	// dt
	float *tmp = dt(dst, LABEL_NUM);

	// truncate and store in destination vector
	minimum += DISC_K;
	for (int value = 0; value < LABEL_NUM; ++value)
		dst[value] = std::min(tmp[value], minimum);

	// normalize
	float val = 0;
	for (int value = 0; value < LABEL_NUM; ++value) 
		val += dst[value];

	val /= LABEL_NUM;
	for (int value = 0; value < LABEL_NUM; ++value) 
		dst[value] -= val;

	delete tmp;
}

// compute message
void compute_message(float *s1, float *s2, float *s3, float *s4, float *dst, const float DISC_K, const int LABEL_NUM)
{
	// aggregate and find min
	float minimum = std::numeric_limits<float>::max();
	for (int value = 0; value < LABEL_NUM; ++value)
	{
		dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
		if (dst[value] < minimum)
			minimum = dst[value];
	}

	// dt
	dt(dst);

	// truncate 
	minimum += DISC_K;
	for (int value = 0; value < LABEL_NUM; ++value)
		if (minimum < dst[value])
			dst[value] = minimum;

	// normalize
	float val = 0;
	for (int value = 0; value < LABEL_NUM; ++value) 
		val += dst[value];

	val /= LABEL_NUM;
	for (int value = 0; value < LABEL_NUM; ++value) 
		dst[value] -= val;
}

// generate output from current messages
image<uchar> * output(image<float *> *u, image<float *> *d, image<float *> *l, image<float *> *r, image<float *> *data, const int LABEL_NUM)
{
	const int width = data->width();
	const int height = data->height();
	image<uchar> *out = new image<uchar>(width, height);

	for (int y = 1; y < height-1; ++y)
	{
		for (int x = 1; x < width-1; ++x)
		{
			// keep track of best value for current pixel
			int best = 0;
			float best_val = std::numeric_limits<float>::max();
			for (int value = 0; value < LABEL_NUM; ++value)
			{
				float val = imRef(u, x, y+1)[value] + imRef(d, x, y-1)[value] + imRef(l, x+1, y)[value] + imRef(r, x-1, y)[value] + imRef(data, x, y)[value];
				if (val < best_val)
				{
					best_val = val;
					best = value;
				}
			}

			imRef(out, x, y) = best;
		}
	}

	return out;
}

// belief propagation using checkerboard update scheme
void bp_cb(image<float *> *u, image<float *> *d, image<float *> *l, image<float *> *r, image<float *> *data, const float DISC_K, const int LABEL_NUM, const int MAX_ITER)
{
	const int width = data->width();  
	const int height = data->height();

	for (int t = 0; t < MAX_ITER; ++t)
		for (int y = 1; y < height - 1; ++y)
			for (int x = ((y+t) % 2) + 1; x < width - 1; x += 2)
			{
				compute_message(imRef(u, x, y+1), imRef(l, x+1, y), imRef(r, x-1, y), imRef(data, x, y), imRef(u, x, y), DISC_K, LABEL_NUM);
				compute_message(imRef(d, x, y-1), imRef(l, x+1, y), imRef(r, x-1, y), imRef(data, x, y), imRef(d, x, y), DISC_K, LABEL_NUM);
				compute_message(imRef(u, x, y+1), imRef(d, x, y-1), imRef(r, x-1, y), imRef(data, x, y), imRef(r, x, y), DISC_K, LABEL_NUM);
				compute_message(imRef(u, x, y+1), imRef(d, x, y-1), imRef(l, x+1, y), imRef(data, x, y), imRef(l, x, y), DISC_K, LABEL_NUM);
			}
}

// multiscale belief propagation
image<uchar> * multiscale_belief_propagation(image<float *> *data0, const float DISC_K, const int LABEL_NUM, const int LEVEL_NUM, const int MAX_ITER)
{
	std::vector<image<float *> *> u(LEVEL_NUM, NULL);
	std::vector<image<float *> *> d(LEVEL_NUM, NULL);
	std::vector<image<float *> *> l(LEVEL_NUM, NULL);
	std::vector<image<float *> *> r(LEVEL_NUM, NULL);
	std::vector<image<float *> *> data(LEVEL_NUM, NULL);

	// data costs
	data[0] = data0;

	// data pyramid
	for (int i = 1; i < LEVEL_NUM; ++i)
	{
		const int old_width = data[i-1]->width();
		const int old_height = data[i-1]->height();
		const int new_width = (int)std::ceil(0.5 * old_width);
		const int new_height = (int)std::ceil(0.5 * old_height);

		assert(new_width >= 1);
		assert(new_height >= 1);

		data[i] = new image<float *>(new_width, new_height);
		for (int y = 0; y < old_height; ++y)
			for (int x = 0; x < old_width; ++x)
				for (int value = 0; value < LABEL_NUM; ++value)
					imRef(data[i], x/2, y/2)[value] += imRef(data[i-1], x, y)[value];
	}

	// run BP from coarse to fine
	for (int i = LEVEL_NUM - 1; i >= 0; --i)
	{
		const int width = data[i]->width();
		const int height = data[i]->height();

		// allocate & init memory for messages
		if (LEVEL_NUM - 1 == i)
		{
			// in the coarsest level messages are initialized to zero
			u[i] = new image<float *>(width, height);
			d[i] = new image<float *>(width, height);
			l[i] = new image<float *>(width, height);
			r[i] = new image<float *>(width, height);
		}
		else
		{
			// initialize messages from values of previous level
			u[i] = new image<float *>(width, height, false);
			d[i] = new image<float *>(width, height, false);
			l[i] = new image<float *>(width, height, false);
			r[i] = new image<float *>(width, height, false);

			for (int y = 0; y < height; ++y)
				for (int x = 0; x < width; ++x)
					for (int value = 0; value < LABEL_NUM; ++value)
					{
						imRef(u[i], x, y)[value] = imRef(u[i+1], x/2, y/2)[value];
						imRef(d[i], x, y)[value] = imRef(d[i+1], x/2, y/2)[value];
						imRef(l[i], x, y)[value] = imRef(l[i+1], x/2, y/2)[value];
						imRef(r[i], x, y)[value] = imRef(r[i+1], x/2, y/2)[value];
					}

			// delete old messages and data
			delete u[i+1];
			delete d[i+1];
			delete l[i+1];
			delete r[i+1];
			delete data[i+1];
		} 

		// BP
		bp_cb(u[i], d[i], l[i], r[i], data[i], DISC_K, LABEL_NUM, MAX_ITER);    
	}

	image<uchar> *out = output(u[0], d[0], l[0], r[0], data[0], LABEL_NUM);

	delete u[0];
	delete d[0];
	delete l[0];
	delete r[0];
	delete data[0];

	return out;
}

void image_restoration()
{
	const std::string input_filename("./mrf_data/noisy_penguin.pgm");
	const std::string output_filename("./mrf_data/penguin_result.pgm");

	const int MAX_ITER = 5;  // number of BP iterations at each scale
	const int LEVEL_NUM = 5;  // number of scales

	const float DISC_K = 200.0f;  // truncation of discontinuity cost
	const float DATA_K = 10000.0f;  // truncation of data cost
	const float LAMBDA = 0.05f;  // weighting of data cost

	const int LABEL_NUM = 256;  // number of possible graylevel values

	// load input
	image<uchar> *img = loadPGM((char *)input_filename.c_str());

	// data costs
	image<float *> *data = compute_data_costs(img, LAMBDA, DATA_K, LABEL_NUM);

	// restore
	image<uchar> *out = multiscale_belief_propagation(data, DISC_K, LABEL_NUM, LEVEL_NUM, MAX_ITER);

	// save output
	savePGM(out, (char *)output_filename.c_str());

	delete img;
	delete out;
}

void stereo()
{
	const std::string input_filename1("./mrf_data/tsukuba1.pgm");
	const std::string input_filename2("./mrf_data/tsukuba2.pgm");
	const std::string output_filename("./mrf_data/stereo_result.pgm");

	const int MAX_ITER = 5;  // number of BP iterations at each scale
	const int LEVEL_NUM = 5;  // number of scales

	const float DISC_K = 1.7f;  // truncation of discontinuity cost
	const float DATA_K = 15.0f;  // truncation of data cost
	const float LAMBDA = 0.07f;  // weighting of data cost

	const int LABEL_NUM = 16;  // number of possible disparities
	const int SCALE = 16;  // scaling from disparity to graylevel in output

	const double SIGMA = 0.7;  // amount to smooth the input images

	// load input
	image<uchar> *img1 = loadPGM((char *)input_filename1.c_str());
	image<uchar> *img2 = loadPGM((char *)input_filename2.c_str());

	// data costs
	image<float *> *data = compute_data_costs(img1, img2, LAMBDA, DATA_K, LABEL_NUM, SIGMA);

	// compute disparities
	image<uchar> *out = multiscale_belief_propagation(data, DISC_K, LABEL_NUM, LEVEL_NUM, MAX_ITER);

	{
		const int width = data->width();
		const int height = data->height();

		for (int y = 1; y < height - 1; ++y)
			for (int x = 1; x < width - 1; ++x)
				imRef(out, x, y) *= SCALE;
	}

	// save output
	savePGM(out, (char *)output_filename.c_str());

	delete img1;
	delete img2;
	delete out;
}

}  // namespace local
}  // unnamed namespace

void bp_vision()
{
	local::image_restoration();
	local::stereo();
}
