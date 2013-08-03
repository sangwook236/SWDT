#include "../efficient_graph_based_image_segmentation_lib/segment-image.h"
#include "../efficient_graph_based_image_segmentation_lib/image.h"
#include "../efficient_graph_based_image_segmentation_lib/misc.h"
#include "../efficient_graph_based_image_segmentation_lib/pnmfile.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <set>
#include <map>


namespace {
namespace local {

class universe_using_map
{
public:
	universe_using_map(const std::set<int> &elements);
	~universe_using_map();
	int find(int x);
	bool contain(int x) const;
	void join(int x, int y);
	int size(int x) const;
	int num_sets() const { return num; }

private:
	std::map<int, uni_elt> elts;
	int num;
};

universe_using_map::universe_using_map(const std::set<int> &elements)
	: num(elements.size())
{
	uni_elt elt;
	for (std::set<int>::const_iterator cit = elements.begin(); cit != elements.end(); ++cit)
	{
		elt.rank = 0;
		elt.size = 1;
		elt.p = *cit;

		elts.insert(std::make_pair(*cit, elt));
	}
}

universe_using_map::~universe_using_map()
{
}

int universe_using_map::size(int x) const
{
#if DEBUG || _DEBUG
	std::map<int, uni_elt>::const_iterator cit = elts.find(x);
	if (elts.end() == cit)
	{
		std::cerr << "can't find an element with index, " << x << std::endl;
		return 0;
	}

	return cit->second.size;
#else
	std::map<int, uni_elt>::const_iterator cit = elts.find(x);
	return elts.end() == cit ? 0 : cit->second.size;
#endif
}

int universe_using_map::find(int x)
{
#if DEBUG || _DEBUG
	int y = x;
	std::map<int, uni_elt>::iterator it = elts.find(y);
	if (elts.end() == it)
	{
		std::cerr << "can't find an element with index, " << y << std::endl;
		return -1;
	}
	else
	{
		while (y != elts[y].p)
		{
			y = elts[y].p;

			std::map<int, uni_elt>::iterator it = elts.find(y);
			if (elts.end() == it)
			{
				std::cerr << "can't find an element with index, " << y << std::endl;
				return -1;
			}
		}
		elts[x].p = y;
	}
#else
	int y = x;
	while (y != elts[y].p)
		y = elts[y].p;
	elts[x].p = y;
#endif
	return y;
}

bool universe_using_map::contain(int x) const
{
	return elts.end() != elts.find(x);
}

void universe_using_map::join(int x, int y)
{
#if DEBUG || _DEBUG
	std::map<int, uni_elt>::iterator it = elts.find(x);
	if (elts.end() == it)
	{
		std::cerr << "can't find an element with index, " << x << std::endl;
		return;
	}
	it = elts.find(y);
	if (elts.end() == it)
	{
		std::cerr << "can't find an element with index, " << y << std::endl;
		return;
	}

	if (elts[x].rank > elts[y].rank)
	{
		elts[y].p = x;
		elts[x].size += elts[y].size;
	}
	else
	{
		elts[x].p = y;
		elts[y].size += elts[x].size;
		if (elts[x].rank == elts[y].rank)
			elts[y].rank++;
	}
#else
	if (elts[x].rank > elts[y].rank)
	{
		elts[y].p = x;
		elts[x].size += elts[y].size;
	}
	else
	{
		elts[x].p = y;
		elts[y].size += elts[x].size;
		if (elts[x].rank == elts[y].rank)
			elts[y].rank++;
	}
#endif
	--num;
}

universe_using_map *segment_graph_using_map(const std::set<int> &vertex_set, int num_edges, edge *edges, float k)
{
	// sort edges by weight
	std::sort(edges, edges + num_edges);

	// make a disjoint-set forest
	const int num_vertices = (int)vertex_set.size();
	universe_using_map *u = new universe_using_map(vertex_set);

	// init thresholds
	std::map<int, float> threshold;
	for (std::set<int>::const_iterator cit = vertex_set.begin(); cit != vertex_set.end(); ++cit)
		threshold[*cit] = THRESHOLD(1,k);

	// for each edge, in non-decreasing weight order...
	for (int i = 0; i < num_edges; ++i)
	{
		edge *pedge = &edges[i];

		// components conected by this edge
		int a = u->find(pedge->a);
		int b = u->find(pedge->b);
		if (a != b)
		{
			if ((pedge->w <= threshold[a]) && (pedge->w <= threshold[b]))
			{
				u->join(a, b);
				a = u->find(a);
				threshold[a] = pedge->w + THRESHOLD(u->size(a), k);
			}
		}
	}

	return u;
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
image<rgb> *segment_image_using_map_container(image<rgb> *im, float sigma, float c, int min_size, int *num_ccs)
{
	int width = im->width();
	int height = im->height();

	image<float> *r = new image<float>(width, height);
	image<float> *g = new image<float>(width, height);
	image<float> *b = new image<float>(width, height);

	// smooth each color channel
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			imRef(r, x, y) = imRef(im, x, y).r;
			imRef(g, x, y) = imRef(im, x, y).g;
			imRef(b, x, y) = imRef(im, x, y).b;
		}
	}
	image<float> *smooth_r = smooth(r, sigma);
	image<float> *smooth_g = smooth(g, sigma);
	image<float> *smooth_b = smooth(b, sigma);
	delete r;
	delete g;
	delete b;

	// build graph
	edge *edges = new edge[width*height*4];
	std::set<int> vertex_set;
	int num = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (x < width-1)
			{
				edges[num].a = y * width + x;
				edges[num].b = y * width + (x+1);
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
				vertex_set.insert(edges[num].a);
				vertex_set.insert(edges[num].b);
				++num;
			}

			if (y < height-1)
			{
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + x;
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
				vertex_set.insert(edges[num].a);
				vertex_set.insert(edges[num].b);
				++num;
			}

			if ((x < width-1) && (y < height-1))
			{
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + (x+1);
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
				vertex_set.insert(edges[num].a);
				vertex_set.insert(edges[num].b);
				++num;
			}

			if ((x < width-1) && (y > 0))
			{
				edges[num].a = y * width + x;
				edges[num].b = (y-1) * width + (x+1);
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
				vertex_set.insert(edges[num].a);
				vertex_set.insert(edges[num].b);
				++num;
			}
		}
	}
	delete smooth_r;
	delete smooth_g;
	delete smooth_b;

	// segment
	universe_using_map *u = segment_graph_using_map(vertex_set, num, edges, c);

	// post process small components
	for (int i = 0; i < num; i++)
	{
		const int a = u->find(edges[i].a);
		const int b = u->find(edges[i].b);
		if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
			u->join(a, b);
	}
	delete [] edges;
	*num_ccs = u->num_sets();

	image<rgb> *output = new image<rgb>(width, height);
	rgb black;
	black.r = 0;
	black.g = 0;
	black.b = 0;

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
		colors[i] = random_rgb();

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (u->contain(y * width + x))
			{
				int comp = u->find(y * width + x);
				imRef(output, x, y) = colors[comp];
			}
			else
			{
				imRef(output, x, y) = black;
			}
		}
	}

	delete [] colors;
	delete u;

	return output;
}

image<rgb> *segment_image_using_depth_guided_map_plus_map_container(image<rgb> *im, image<rgb> *depth, float sigma, float c, int min_size, int *num_ccs)
{
	int width = im->width();
	int height = im->height();

	image<float> *r = new image<float>(width, height);
	image<float> *g = new image<float>(width, height);
	image<float> *b = new image<float>(width, height);

	// smooth each color channel
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			imRef(r, x, y) = imRef(im, x, y).r;
			imRef(g, x, y) = imRef(im, x, y).g;
			imRef(b, x, y) = imRef(im, x, y).b;
		}
	}
	image<float> *smooth_r = smooth(r, sigma);
	image<float> *smooth_g = smooth(g, sigma);
	image<float> *smooth_b = smooth(b, sigma);
	delete r;
	delete g;
	delete b;

	// build graph
	edge *edges = new edge[width*height*4];
	std::set<int> vertex_set;
	int num = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const uchar sa = imRef(depth, x, y).r;
			if (255 != sa) continue;

			if (x < width-1)
			{
				const uchar sb = imRef(depth, x+1, y).r;
				if (255 == sb)  // for pixels in valid depth regions.
				{
					edges[num].a = y * width + x;
					edges[num].b = y * width + (x+1);
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
					vertex_set.insert(edges[num].a);
					vertex_set.insert(edges[num].b);
					++num;
				}
			}

			if (y < height-1)
			{
				const uchar sb = imRef(depth, x, y+1).r;
				if (255 == sb)  // for pixels in valid depth regions.
				{
					edges[num].a = y * width + x;
					edges[num].b = (y+1) * width + x;
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
					vertex_set.insert(edges[num].a);
					vertex_set.insert(edges[num].b);
					++num;
				}
			}

			if ((x < width-1) && (y < height-1))
			{
				const uchar sb = imRef(depth, x+1, y+1).r;
				if (255 == sb)  // for pixels in valid depth regions.
				{
					edges[num].a = y * width + x;
					edges[num].b = (y+1) * width + (x+1);
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
					vertex_set.insert(edges[num].a);
					vertex_set.insert(edges[num].b);
					++num;
				}
			}

			if ((x < width-1) && (y > 0))
			{
				const uchar sb = imRef(depth, x+1, y-1).r;
				if (255 == sb)  // for pixels in valid depth regions.
				{
					edges[num].a = y * width + x;
					edges[num].b = (y-1) * width + (x+1);
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
					vertex_set.insert(edges[num].a);
					vertex_set.insert(edges[num].b);
					++num;
				}
			}
		}
	}
	delete smooth_r;
	delete smooth_g;
	delete smooth_b;

	// segment
	universe_using_map *u = segment_graph_using_map(vertex_set, num, edges, c);

	// post process small components
	for (int i = 0; i < num; i++)
	{
		const int a = u->find(edges[i].a);
		const int b = u->find(edges[i].b);
		if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
			u->join(a, b);
	}
	delete [] edges;
	*num_ccs = u->num_sets();

	image<rgb> *output = new image<rgb>(width, height);
	rgb black;
	black.r = 0;
	black.g = 0;
	black.b = 0;

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
		colors[i] = random_rgb();

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (u->contain(y * width + x))
			{
				int comp = u->find(y * width + x);
				imRef(output, x, y) = colors[comp];
			}
			else
			{
				imRef(output, x, y) = black;
			}
		}
	}

	delete [] colors;
	delete u;

	return output;
}

image<rgb> * segment_image_using_depth_guided_map(image<rgb> *im, image<rgb> *depth, float sigma, float c, int min_size, int *num_ccs)
{
	const int width = im->width();
	const int height = im->height();

	image<float> *r = new image<float>(width, height);
	image<float> *g = new image<float>(width, height);
	image<float> *b = new image<float>(width, height);

	// smooth each color channel
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			imRef(r, x, y) = imRef(im, x, y).r;
			imRef(g, x, y) = imRef(im, x, y).g;
			imRef(b, x, y) = imRef(im, x, y).b;
		}
	}
	image<float> *smooth_r = smooth(r, sigma);
	image<float> *smooth_g = smooth(g, sigma);
	image<float> *smooth_b = smooth(b, sigma);
	delete r;
	delete g;
	delete b;

	// build graph
	edge *edges = new edge[width * height * 4];
	int num = 0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const uchar sa = imRef(depth, x, y).r;

			if (x < width-1)
			{
				const uchar sb = imRef(depth, x+1, y).r;

				edges[num].a = y * width + x;
				edges[num].b = y * width + (x+1);
				if (255 == sa && 255 == sb)
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
				else if (255 == sa || 255 == sb)
					edges[num].w = std::numeric_limits<float>::max();
				else
					edges[num].w = 0.0f;
				++num;
			}

			if (y < height-1)
			{
				const uchar sb = imRef(depth, x, y+1).r;

				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + x;
				if (255 == sa && 255 == sb)
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
				else if (255 == sa || 255 == sb)
					edges[num].w = std::numeric_limits<float>::max();
				else
					edges[num].w = 0.0f;
				++num;
			}

			if ((x < width-1) && (y < height-1))
			{
				const uchar sb = imRef(depth, x+1, y+1).r;

				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + (x+1);
				if (255 == sa && 255 == sb)
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
				else if (255 == sa || 255 == sb)
					edges[num].w = std::numeric_limits<float>::max();
				else
					edges[num].w = 0.0f;
				++num;
			}

			if ((x < width-1) && (y > 0))
			{
				const uchar sb = imRef(depth, x+1, y-1).r;

				edges[num].a = y * width + x;
				edges[num].b = (y-1) * width + (x+1);
				if (255 == sa && 255 == sb)
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
				else if (255 == sa || 255 == sb)
					edges[num].w = std::numeric_limits<float>::max();
				else
					edges[num].w = 0.0f;
				++num;
			}
		}
	}
	delete smooth_r;
	delete smooth_g;
	delete smooth_b;

	// segment
	universe *u = segment_graph(width*height, num, edges, c);

	// post process small components
	for (int i = 0; i < num; ++i)
	{
		const int a = u->find(edges[i].a);
		const int b = u->find(edges[i].b);
		if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
			u->join(a, b);
	}
	delete [] edges;
	*num_ccs = u->num_sets();

	image<rgb> *output = new image<rgb>(width, height);

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; ++i)
		colors[i] = random_rgb();

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int comp = u->find(y * width + x);
			imRef(output, x, y) = colors[comp];
		}
	}

	delete [] colors;
	delete u;

	return output;
}

/*
	The program takes a color image (PPM format) and produces a segmentation
	with a random color assigned to each region.

	run "segment sigma k min input output".

	The parameters are: (see the paper for details)

	sigma: Used to smooth the input image before segmenting it.
	k: Value for the threshold function.
	min: Minimum component size enforced by post-processing.
	input: Input image.
	output: Output image.
*/

// [ref] ${Efficient_Graph_Based_Image_Segmentation_HOME}/segment.cpp
void sample(const bool use_map_container)
{
#if 0
	const std::string input_filename("./data/search_algorithm/beach.ppm");
	const std::string output_filename("./data/search_algorithm/beach_segmented.ppm");
	const float sigma = 0.5f;
	const float k = 500.0f;
	const int min_size = 50;
#elif 0
	const std::string input_filename("./data/search_algorithm/grain.ppm");
	const std::string output_filename("./data/search_algorithm/grain_segmented.ppm");
	const float sigma = 0.5f;
	const float k = 1000.0f;
	const int min_size = 100;
#elif 1
	//const std::string input_filename("./data/search_algorithm/kinect_rgba_20130530T103805.ppm");
	//const std::string input_filename("./data/search_algorithm/kinect_rgba_20130531T023152.ppm");
	//const std::string input_filename("./data/search_algorithm/kinect_rgba_20130531T023346.ppm");
	//const std::string input_filename("./data/search_algorithm/kinect_rgba_20130531T023359.ppm");
	const std::string input_filename("./data/search_algorithm/rectified_image_rgb_0.ppm");
	const std::string output_filename("./data/search_algorithm/kinect_segmented.ppm");
	const float sigma = 0.5f;
	const float k = 500.0f;
	const int min_size = 50;
#endif

	std::cout << "loading input image." << std::endl;
	image<rgb> *input = loadPPM(input_filename.c_str());  // color

	std::cout << "processing" << std::endl;

	image<rgb> *seg = NULL;
	int num_ccs = 0;
	{
		boost::timer::auto_cpu_timer timer;

		if (use_map_container)
			seg = segment_image_using_map_container(input, sigma, k, min_size, &num_ccs);
		else  // original implementation.
			seg = segment_image(input, sigma, k, min_size, &num_ccs);
	}

	savePPM(seg, output_filename.c_str());

	std::cout << "got " << num_ccs << " components" << std::endl;

	// show results
	cv::Mat img(seg->height(), seg->width(), CV_8UC3, (void *)seg->data);

#if 1
	cv::imshow("segmented image", img);
#else
	double minVal, maxVal;
	cv::minMaxLoc(img, &minVal, &maxVal);
	cv::Mat tmp;
	img.convertTo(tmp, CV_8UC3, 255.0 / maxVal, 0.0);

	cv::imshow("segmented image", tmp);
#endif

	cv::waitKey(0);

	cv::destroyAllWindows();

	delete seg;
	seg = NULL;
}

void sample_using_depth_guided_map(const bool use_map_container)
{
#if 1
	const std::string input_filename("./data/search_algorithm/rectified_image_rgb_0.ppm");
	const std::string depth_filename("./data/search_algorithm/depth_guided_mask_0.ppm");
	const std::string output_filename("./data/search_algorithm/segmented_image_rgb_0.ppm");
#elif 0
	const std::string input_filename("./data/search_algorithm/rectified_image_rgb_1.ppm");
	const std::string depth_filename("./data/search_algorithm/depth_guided_mask_1.ppm");
	const std::string output_filename("./data/search_algorithm/segmented_image_rgb_1.ppm");
#elif 0
	const std::string input_filename("./data/search_algorithm/rectified_image_rgb_2.ppm");
	const std::string depth_filename("./data/search_algorithm/depth_guided_mask_2.ppm");
	const std::string output_filename("./data/search_algorithm/segmented_image_rgb_2.ppm");
#elif 0
	const std::string input_filename("./data/search_algorithm/rectified_image_rgb_3.ppm");
	const std::string depth_filename("./data/search_algorithm/depth_guided_mask_3.ppm");
	const std::string output_filename("./data/search_algorithm/segmented_image_rgb_3.ppm");
#endif
	const float sigma = 0.5f;
	const float k = 500.0f;
	const int min_size = 50;

	std::cout << "loading input image." << std::endl;
	image<rgb> *input = loadPPM(input_filename.c_str());  // RGB
	image<rgb> *depth = loadPPM(depth_filename.c_str());  // depth

	std::cout << "processing" << std::endl;

	image<rgb> *seg = NULL;
	int num_ccs = 0;
	{
		boost::timer::auto_cpu_timer timer;

		if (use_map_container)
			seg = segment_image_using_depth_guided_map_plus_map_container(input, depth, sigma, k, min_size, &num_ccs);
		else  // original implementation.
			seg = segment_image_using_depth_guided_map(input, depth, sigma, k, min_size, &num_ccs);
	}

	savePPM(seg, output_filename.c_str());

	std::cout << "got " << num_ccs << " components" << std::endl;

	// show results
	cv::Mat img(seg->height(), seg->width(), CV_8UC3, (void *)seg->data);

#if 1
	cv::imshow("segmented image", img);
#else
	double minVal, maxVal;
	cv::minMaxLoc(img, &minVal, &maxVal);
	cv::Mat tmp;
	img.convertTo(tmp, CV_8UC3, 255.0 / maxVal, 0.0);

	cv::imshow("segmented image", tmp);
#endif

	cv::waitKey(0);

	cv::destroyAllWindows();

	delete seg;
	seg = NULL;
}

}  // namespace local
}  // unnamed namespace

namespace my_efficient_graph_based_image_segmentation {

}  // namespace my_efficient_graph_based_image_segmentation

/*
[ref]
	"Efficient Graph-Based Image Segmentation", Pedro F. Felzenszwalb and Daniel P. Huttenlocher, IJCV, 2004.
	http://cs.brown.edu/~pff/segment/
*/

int efficient_graph_based_image_segmentation_main(int argc, char *argv[])
{
	const bool use_map_container = false;

	//local::sample(use_map_container);
	local::sample_using_depth_guided_map(use_map_container);

	return 0;
}
