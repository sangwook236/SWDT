#include "../bp_vision_lib/image.h"
#include "../bp_vision_lib/misc.h"
#include "../bp_vision_lib/pnmfile.h"
#include "../bp_vision_lib/filter.h"
#include "../bp_vision_lib/imconv.h"
#include <boost/multi_array.hpp>
#include <boost/smart_ptr.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>


namespace {
namespace local {

// width x height x label
typedef boost::multi_array<float, 3> label_data_type;
typedef label_data_type::array_view<1>::type label_data_view_type;
typedef void (*compute_message_pfn_type)(const label_data_view_type &s1, const label_data_view_type &s2, const label_data_view_type &s3, const label_data_view_type &s4, label_data_view_type &dst, const float DISC_K, const int LABEL_NUM);

void clear_label_data(label_data_type &data)
{
    memset(data.data(), 0, data.num_elements() * sizeof(float));
}

void copy_label_data(label_data_type &dst, const label_data_type &src)
{
    memcpy(dst.data(), src.data(), dst.num_elements() * sizeof(float));
}

// computation of data costs (for image restoration)
boost::shared_ptr<label_data_type> compute_data_costs(image<uchar> *img, const float LAMBDA, const float DATA_K, const int LABEL_NUM)
{
	const int width = img->width();
	const int height = img->height();
	boost::shared_ptr<label_data_type> data(new label_data_type(boost::extents[width][height][LABEL_NUM]));

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
			for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
			{
				const float val = square(float(imRef(img, x, y) - lbl));
				(*data)[x][y][lbl] = LAMBDA * std::min(val, DATA_K);
			}

	return data;
}

// computation of data costs (for stereo)
boost::shared_ptr<label_data_type> compute_data_costs(image<uchar> *img1, image<uchar> *img2, const float LAMBDA, const float DATA_K, const int LABEL_NUM, const double SIGMA)
{
	const int width = img1->width();
	const int height = img1->height();
	boost::shared_ptr<label_data_type> data(new label_data_type(boost::extents[width][height][LABEL_NUM]));

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
			for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
			{
				const float val = std::abs(imRef(sm1, x, y) - imRef(sm2, x - lbl, y));
				(*data)[x][y][lbl] = LAMBDA * std::min(val, DATA_K);
			}

	delete sm1;
	delete sm2;

	return data;
}

// distance transform (dt) of 1d function (for image restoration)
float * dt_quadratic(label_data_view_type &f, int LABEL_NUM)
{
	int *v = new int [LABEL_NUM];
	float *z = new float [LABEL_NUM + 1];
	v[0] = 0;
	z[0] = -std::numeric_limits<float>::max();
	z[1] = +std::numeric_limits<float>::max();

	int k = 0;
	for (int q = 1; q <= LABEL_NUM - 1; ++q)
	{
		float s = ((f[q] + square(q)) - (f[v[k]] + square(v[k]))) / (2 * (q - v[k]));
		while (s <= z[k])
		{
			--k;
			s = ((f[q] + square(q)) - (f[v[k]] + square(v[k]))) / (2 * ( q - v[k]));
		}
		++k;
		v[k] = q;
		z[k] = s;
		z[k+1] = +std::numeric_limits<float>::max();
	}

	float *d = new float [LABEL_NUM];
	k = 0;
	for (int q = 0; q <= LABEL_NUM - 1; ++q)
	{
		while (z[k+1] < q)
			++k;
		d[q] = square(q - v[k]) + f[v[k]];
	}

	delete [] v;
	delete [] z;

	return d;
}

// distance transform (dt) of 1d function (for stereo)
void dt_linear(label_data_view_type &f, const int LABEL_NUM)
{
	for (int q = 1; q < LABEL_NUM; ++q)
	{
		float prev = f[q-1] + 1.0F;
		if (prev < f[q])
			f[q] = prev;
	}
	for (int q = LABEL_NUM - 2; q >= 0; --q)
	{
		float prev = f[q+1] + 1.0F;
		if (prev < f[q])
			f[q] = prev;
	}
}

// compute message (for image restoration)
void compute_message_quadratic(const label_data_view_type &s1, const label_data_view_type &s2, const label_data_view_type &s3, const label_data_view_type &s4, label_data_view_type &dst, const float DISC_K, const int LABEL_NUM)
{
	// aggregate and find min
	float minimum = std::numeric_limits<float>::max();
	for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
	{
		dst[lbl] = s1[lbl] + s2[lbl] + s3[lbl] + s4[lbl];
		if (dst[lbl] < minimum)
			minimum = dst[lbl];
	}

	// dt
	float *tmp = dt_quadratic(dst, LABEL_NUM);

	// truncate and store in destination vector
	minimum += DISC_K;
	for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
		dst[lbl] = std::min(tmp[lbl], minimum);

	// normalize
	float val = 0;
	for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
		val += dst[lbl];

	val /= LABEL_NUM;
	for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
		dst[lbl] -= val;

	delete [] tmp;
}

// compute message (for stereo)
void compute_message_linear(const label_data_view_type &s1, const label_data_view_type &s2, const label_data_view_type &s3, const label_data_view_type &s4, label_data_view_type &dst, const float DISC_K, const int LABEL_NUM)
{
	// aggregate and find min
	float minimum = std::numeric_limits<float>::max();
	for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
	{
		dst[lbl] = s1[lbl] + s2[lbl] + s3[lbl] + s4[lbl];
		if (dst[lbl] < minimum)
			minimum = dst[lbl];
	}

	// dt
	dt_linear(dst, LABEL_NUM);

	// truncate
	minimum += DISC_K;
	for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
		if (minimum < dst[lbl])
			dst[lbl] = minimum;

	// normalize
	float val = 0;
	for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
		val += dst[lbl];

	val /= LABEL_NUM;
	for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
		dst[lbl] -= val;
}

// generate output from current messages
image<uchar> * output(label_data_type &u, label_data_type &d, label_data_type &l, label_data_type &r, label_data_type &data, const int LABEL_NUM)
{
	const int width = data.shape()[0];
	const int height = data.shape()[1];
	image<uchar> *out = new image<uchar>(width, height);

	for (int y = 1; y < height - 1; ++y)
	{
		for (int x = 1; x < width - 1; ++x)
		{
			// keep track of best label for current pixel
			int best = 0;
			float best_val = std::numeric_limits<float>::max();
			for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
			{
				float val = u[x][y+1][lbl] + d[x][y-1][lbl] + l[x+1][y][lbl] + r[x-1][y][lbl] + data[x][y][lbl];
				if (val < best_val)
				{
					best_val = val;
					best = lbl;
				}
			}

			imRef(out, x, y) = best;
		}
	}

	return out;
}

// belief propagation using checkerboard update scheme
void bp_cb(label_data_type &u, label_data_type &d, label_data_type &l, label_data_type &r, label_data_type &data, const float DISC_K, const int LABEL_NUM, const int MAX_ITER, compute_message_pfn_type compute_message)
{
	const int width = data.shape()[0];
	const int height = data.shape()[1];

	for (int iter = 0; iter < MAX_ITER; ++iter)
	{
	    std::cout << "iter " << iter << std::endl;
		for (int y = 1; y < height - 1; ++y)
			for (int x = ((y+iter) % 2) + 1; x < width - 1; x += 2)
			{
#if 0
				(*compute_message)(u[boost::indices[x][y+1][label_data_type::index_range()]], l[boost::indices[x+1][y][label_data_type::index_range()]], r[boost::indices[x-1][y][label_data_type::index_range()]], data[boost::indices[x][y][label_data_type::index_range()]], u[boost::indices[x][y][label_data_type::index_range()]], DISC_K, LABEL_NUM);
				(*compute_message)(d[boost::indices[x][y-1][label_data_type::index_range()]], l[boost::indices[x+1][y][label_data_type::index_range()]], r[boost::indices[x-1][y][label_data_type::index_range()]], data[boost::indices[x][y][label_data_type::index_range()]], d[boost::indices[x][y][label_data_type::index_range()]], DISC_K, LABEL_NUM);
				(*compute_message)(u[boost::indices[x][y+1][label_data_type::index_range()]], d[boost::indices[x][y-1][label_data_type::index_range()]], r[boost::indices[x-1][y][label_data_type::index_range()]], data[boost::indices[x][y][label_data_type::index_range()]], r[boost::indices[x][y][label_data_type::index_range()]], DISC_K, LABEL_NUM);
				(*compute_message)(u[boost::indices[x][y+1][label_data_type::index_range()]], d[boost::indices[x][y-1][label_data_type::index_range()]], l[boost::indices[x+1][y][label_data_type::index_range()]], data[boost::indices[x][y][label_data_type::index_range()]], l[boost::indices[x][y][label_data_type::index_range()]], DISC_K, LABEL_NUM);
#else
				{
					label_data_view_type aa = u[boost::indices[x][y+1][label_data_type::index_range()]], bb = l[boost::indices[x+1][y][label_data_type::index_range()]], cc = r[boost::indices[x-1][y][label_data_type::index_range()]], dd = data[boost::indices[x][y][label_data_type::index_range()]], ee = u[boost::indices[x][y][label_data_type::index_range()]];
					(*compute_message)(aa, bb, cc, dd, ee, DISC_K, LABEL_NUM);
				}
				{
					label_data_view_type aa = d[boost::indices[x][y-1][label_data_type::index_range()]], bb = l[boost::indices[x+1][y][label_data_type::index_range()]], cc = r[boost::indices[x-1][y][label_data_type::index_range()]], dd = data[boost::indices[x][y][label_data_type::index_range()]], ee = d[boost::indices[x][y][label_data_type::index_range()]];
					(*compute_message)(aa, bb, cc, dd, ee, DISC_K, LABEL_NUM);
				}
				{
					label_data_view_type aa = u[boost::indices[x][y+1][label_data_type::index_range()]], bb = d[boost::indices[x][y-1][label_data_type::index_range()]], cc = r[boost::indices[x-1][y][label_data_type::index_range()]], dd = data[boost::indices[x][y][label_data_type::index_range()]], ee = r[boost::indices[x][y][label_data_type::index_range()]];
					(*compute_message)(aa, bb, cc, dd, ee, DISC_K, LABEL_NUM);
				}
				{
					label_data_view_type aa = u[boost::indices[x][y+1][label_data_type::index_range()]], bb = d[boost::indices[x][y-1][label_data_type::index_range()]], cc = l[boost::indices[x+1][y][label_data_type::index_range()]], dd = data[boost::indices[x][y][label_data_type::index_range()]], ee = l[boost::indices[x][y][label_data_type::index_range()]];
					(*compute_message)(aa, bb, cc, dd, ee, DISC_K, LABEL_NUM);
				}
#endif
			}
	}
}

// multiscale belief propagation
image<uchar> * multiscale_belief_propagation(const label_data_type &data0, const float DISC_K, const int LABEL_NUM, const int LEVEL_NUM, const int MAX_ITER, compute_message_pfn_type compute_message)
{
	std::vector<label_data_type> u(LEVEL_NUM);
	std::vector<label_data_type> d(LEVEL_NUM);
	std::vector<label_data_type> l(LEVEL_NUM);
	std::vector<label_data_type> r(LEVEL_NUM);
	std::vector<label_data_type> data(LEVEL_NUM);

	// data costs
	//data[0] = data0;  // run-time error: ???
	data[0].resize(boost::extents[data0.shape()[0]][data0.shape()[1]][LABEL_NUM]);
	//data[0].assign(data0.begin(), data0.end());  // compile-time error: ???
	copy_label_data(data[0], data0);

	// data pyramid
	for (int lvl = 1; lvl < LEVEL_NUM; ++lvl)
	{
		const int old_width = data[lvl-1].shape()[0];
		const int old_height = data[lvl-1].shape()[1];
		const int new_width = (int)std::ceil(0.5 * old_width);
		const int new_height = (int)std::ceil(0.5 * old_height);

		assert(new_width >= 1);
		assert(new_height >= 1);

		data[lvl].resize(boost::extents[new_width][new_height][LABEL_NUM]);
		for (int y = 0; y < old_height; ++y)
			for (int x = 0; x < old_width; ++x)
				for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
					data[lvl][x/2][y/2][lbl] += data[lvl-1][x][y][lbl];
	}

	// run BP from coarse to fine
	for (int lvl = LEVEL_NUM - 1; lvl >= 0; --lvl)
	{
		const int width = data[lvl].shape()[0];
		const int height = data[lvl].shape()[1];

		u[lvl].resize(boost::extents[width][height][LABEL_NUM]);
		d[lvl].resize(boost::extents[width][height][LABEL_NUM]);
		l[lvl].resize(boost::extents[width][height][LABEL_NUM]);
		r[lvl].resize(boost::extents[width][height][LABEL_NUM]);

		// allocate & init memory for messages
		if (LEVEL_NUM - 1 == lvl)
		{
			// in the coarsest level messages are initialized to zero
			clear_label_data(u[lvl]);
			clear_label_data(d[lvl]);
			clear_label_data(l[lvl]);
			clear_label_data(r[lvl]);
		}
		else
		{
			// initialize messages from values of previous level
			for (int y = 0; y < height; ++y)
				for (int x = 0; x < width; ++x)
					for (int lbl = 0; lbl < LABEL_NUM; ++lbl)
					{
						u[lvl][x][y][lbl] = u[lvl+1][x/2][y/2][lbl];
						d[lvl][x][y][lbl] = d[lvl+1][x/2][y/2][lbl];
						l[lvl][x][y][lbl] = l[lvl+1][x/2][y/2][lbl];
						r[lvl][x][y][lbl] = r[lvl+1][x/2][y/2][lbl];
					}

			// delete old messages and data
			u[lvl+1].resize(boost::extents[0][0][0]);
			d[lvl+1].resize(boost::extents[0][0][0]);
			l[lvl+1].resize(boost::extents[0][0][0]);
			r[lvl+1].resize(boost::extents[0][0][0]);
			data[lvl+1].resize(boost::extents[0][0][0]);
		}

		// BP
		bp_cb(u[lvl], d[lvl], l[lvl], r[lvl], data[lvl], DISC_K, LABEL_NUM, MAX_ITER, compute_message);
	}

	image<uchar> *out = output(u[0], d[0], l[0], r[0], data[0], LABEL_NUM);

	u[0].resize(boost::extents[0][0][0]);
	d[0].resize(boost::extents[0][0][0]);
	l[0].resize(boost::extents[0][0][0]);
	r[0].resize(boost::extents[0][0][0]);
	data[0].resize(boost::extents[0][0][0]);

	return out;
}

void image_restoration()
{
	const std::string input_filename("./probabilistic_graphical_model_data/bp_vision/noisy_penguin.pgm");
	const std::string output_filename("./probabilistic_graphical_model_data/bp_vision/penguin_result.pgm");

	const int MAX_ITER = 5;  // number of BP iterations at each scale
	const int LEVEL_NUM = 5;  // number of scales

	const float DISC_K = 200.0f;  // truncation of discontinuity cost
	const float DATA_K = 10000.0f;  // truncation of data cost
	const float LAMBDA = 0.05f;  // weighting of data cost

	const int LABEL_NUM = 256;  // number of possible graylevel values

	// load input
	image<uchar> *img = loadPGM((char *)input_filename.c_str());

	// data costs
#if defined(__GNUC__)
	boost::shared_ptr<label_data_type> data(compute_data_costs(img, LAMBDA, DATA_K, LABEL_NUM));
#else
	boost::shared_ptr<label_data_type> &data = compute_data_costs(img, LAMBDA, DATA_K, LABEL_NUM);
#endif

	// restore
	image<uchar> *out = multiscale_belief_propagation(*data, DISC_K, LABEL_NUM, LEVEL_NUM, MAX_ITER, &compute_message_quadratic);

	// save output
	savePGM(out, (char *)output_filename.c_str());

	delete img;
	delete out;
}

void stereo()
{
	const std::string input_filename1("./probabilistic_graphical_model_data/bp_vision/tsukuba1.pgm");
	const std::string input_filename2("./probabilistic_graphical_model_data/bp_vision/tsukuba2.pgm");
	const std::string output_filename("./probabilistic_graphical_model_data/bp_vision/stereo_result.pgm");

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
#if defined(__GNUC__)
	boost::shared_ptr<label_data_type> data(compute_data_costs(img1, img2, LAMBDA, DATA_K, LABEL_NUM, SIGMA));
#else
	boost::shared_ptr<label_data_type> &data = compute_data_costs(img1, img2, LAMBDA, DATA_K, LABEL_NUM, SIGMA);
#endif

	// compute disparities
	image<uchar> *out = multiscale_belief_propagation(*data, DISC_K, LABEL_NUM, LEVEL_NUM, MAX_ITER, &compute_message_linear);

	{
		const int width = data->shape()[0];
		const int height = data->shape()[1];

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

namespace my_bp_vision {

}  // namespace my_bp_vision

int bp_vision_main(int argc, char *argv[])
{
	//local::image_restoration();
	local::stereo();

	return 0;
}
