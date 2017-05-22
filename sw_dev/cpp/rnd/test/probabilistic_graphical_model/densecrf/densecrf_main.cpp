#include <densecrf/densecrf.h>
#include <densecrf/optimization.h>
#include <opencv2/opencv.hpp>
#include <boost/timer.hpp>
#include <iostream>
#include <cmath>
#include <memory>


namespace {
namespace local {

// REF [file] >> ${DENSECRF_HOME}/examples/common.cpp

// Store the colors we read, so that we can write them again.
int nColors = 0;
int colors[255];

int getColor(const unsigned char *c)
{
	return c[0] + 256 * c[1] + 256 * 256 * c[2];
}

void putColor(unsigned char *c, int cc)
{
	c[0] = cc & 0xff;
	c[1] = (cc >> 8) & 0xff;
	c[2] = (cc >> 16) & 0xff;
}

// Produce a color image from a bunch of labels.
unsigned char * colorize(const VectorXs &labeling, int W, int H)
{
	unsigned char *r = new unsigned char[W * H * 3];
	for (int k = 0; k < W * H; ++k)
	{
		const int c = colors[labeling[k]];
		putColor(r + 3 * k, c);
	}
	//std::cout << r[0] << ' ' << r[1] << ' ' << r[2] << ' ' << std::end;
	return r;
}

// Read the labeling from a file.
VectorXs getLabeling(const unsigned char *im, int N, int M)
{
	VectorXs res(N);
	//std::cout << im[0] << ' ' << im[1] << ' ' << im[2] << ' ' << std::end;
	for (int k = 0; k < N; ++k)
	{
		// Map the color to a label.
		int c = getColor(im + 3 * k);
		int i;
		for (i = 0; i<nColors && c != colors[i]; i++);
		if (c && i == nColors)
		{
			if (i < M)
				colors[nColors++] = c;
			else
				c = 0;
		}
		res[k] = c ? i : -1;
	}
	return res;
}

MatrixXf computeUnary(const VectorXs &lbl, const int NUM_LABELS, const float GT_PROB)
{
	const float u_energy = -(float)std::log(1.0 / NUM_LABELS);
	const float n_energy = -(float)std::log((1.0 - GT_PROB) / (NUM_LABELS - 1));
	const float p_energy = -(float)std::log(GT_PROB);

	MatrixXf r(NUM_LABELS, lbl.rows());
	r.fill(u_energy);
	//std::cout << im[0] << ' ' << im[1] << ' ' << im[2] << ' ' << std::end;
	for (int k = 0; k < lbl.rows(); ++k)
	{
		// Set the energy.
		if (lbl[k] >= 0)
		{
			r.col(k).fill(n_energy);
			r(lbl[k], k) = p_energy;
		}
	}
	return r;
}

// The energy object implements an energy function that is minimized using LBFGS.
class CRFEnergy : public EnergyFunction
{
public:
	CRFEnergy(DenseCRF & crf, const ObjectiveFunction & objective, int NIT, bool unary = 1, bool pairwise = 1, bool kernel = 1)
	: crf_(crf), objective_(objective), NIT_(NIT), unary_(unary), pairwise_(pairwise), kernel_(kernel), l2_norm_(0.f)
	{
		initial_u_param_ = crf_.unaryParameters();
		initial_lbl_param_ = crf_.labelCompatibilityParameters();
		initial_knl_param_ = crf_.kernelParameters();
	}

	void setL2Norm(float norm)
	{
		l2_norm_ = norm;
	}

	virtual VectorXf initialValue()
	{
		VectorXf p(unary_*initial_u_param_.rows() + pairwise_*initial_lbl_param_.rows() + kernel_*initial_knl_param_.rows());
		p << (unary_ ? initial_u_param_ : VectorXf()), (pairwise_ ? initial_lbl_param_ : VectorXf()), (kernel_ ? initial_knl_param_ : VectorXf());
		return p;
	}

	virtual double gradient(const VectorXf &x, VectorXf &dx)
	{
		int p = 0;
		if (unary_)
		{
			crf_.setUnaryParameters(x.segment(p, initial_u_param_.rows()));
			p += initial_u_param_.rows();
		}
		if (pairwise_)
		{
			crf_.setLabelCompatibilityParameters(x.segment(p, initial_lbl_param_.rows()));
			p += initial_lbl_param_.rows();
		}
		if (kernel_)
			crf_.setKernelParameters(x.segment(p, initial_knl_param_.rows()));

		VectorXf du = 0 * initial_u_param_, dl = 0 * initial_u_param_, dk = 0 * initial_knl_param_;
		double r = crf_.gradient(NIT_, objective_, unary_ ? &du : NULL, pairwise_ ? &dl : NULL, kernel_ ? &dk : NULL);
		dx.resize(unary_*du.rows() + pairwise_*dl.rows() + kernel_*dk.rows());
		dx << -(unary_ ? du : VectorXf()), -(pairwise_ ? dl : VectorXf()), -(kernel_ ? dk : VectorXf());
		r = -r;
		if (l2_norm_ > 0)
		{
			dx += l2_norm_ * x;
			r += 0.5*l2_norm_ * (x.dot(x));
		}

		return r;
	}

protected:
	VectorXf initial_u_param_, initial_lbl_param_, initial_knl_param_;
	DenseCRF &crf_;
	const ObjectiveFunction &objective_;
	int NIT_;
	bool unary_, pairwise_, kernel_;
	float l2_norm_;
};

// REF [file] >> ${DENSECRF_HOME}/examples/dense_inference.cpp
void inference_example()
{
#if 1
	const std::string img_filename("./data/probabilistic_graphical_model/im1.ppm");
	const std::string anno_filename("./data/probabilistic_graphical_model/anno1.ppm");
#elif 0
	const std::string img_filename("./data/probabilistic_graphical_model/im2.ppm");
	const std::string anno_filename("./data/probabilistic_graphical_model/anno2.ppm");
#elif 0
	const std::string img_filename("./data/probabilistic_graphical_model/im3.ppm");
	const std::string anno_filename("./data/probabilistic_graphical_model/anno3.ppm");
#else
	const std::string img_filename("D:/dataset/phenotyping/RDA/20160406_trimmed_plant/adaptor1/side_0.png");
	const std::string anno_filename("D:/dataset/phenotyping/RDA/20160406_trimmed_plant/adaptor1/side_0.png.init_fg.png");
#endif

	// Number of labels.
	const int NUM_LABELS = 21;

	// Load the color image and some crude annotations (which are used in a simple classifier).
	const cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Failed to load image: " << img_filename << std::endl;
		return;
	}
	const cv::Mat anno(cv::imread(anno_filename, cv::IMREAD_COLOR));
	if (anno.empty())
	{
		std::cerr << "Failed to load annotations: " << anno_filename << std::endl;
		return;
	}
	if (img.rows != anno.rows || img.cols != anno.cols)
	{
		std::cerr << "Annotation size doesn't match image!" << std::endl;
		return;
	}

	// Put your own unary classifier here.
	// Certainty that the groundtruth is correct.
	const float GT_PROB = 0.5;
	// Simple classifier that is 50% certain that the annotation is correct.
	const MatrixXf unary(computeUnary(getLabeling(anno.data, anno.rows * anno.cols, NUM_LABELS), NUM_LABELS, GT_PROB));

	// Setup the CRF model.
	DenseCRF2D crf(img.cols, img.rows, NUM_LABELS);
	// Specify the unary potential as an array of size W*H*(#classes).
	//	Packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
	crf.setUnaryEnergy(unary);
	// Add a color independent term (feature = pixel location 0..W-1, 0..H-1).
	//	x_stddev = 3.
	//	y_stddev = 3.
	//	weight = 3.
	crf.addPairwiseGaussian(3, 3, new PottsCompatibility(3));
	// Add a color dependent term (feature = xyrgb).
	//	x_stddev = 60.
	//	y_stddev = 60.
	//	r_stddev = g_stddev = b_stddev = 20.
	//	weight = 10.
	crf.addPairwiseBilateral(80, 80, 13, 13, 13, img.data, new PottsCompatibility(10));

	// Do MAP inference.
#if 0
	MatrixXf Q = crf.startInference(), t1, t2;
	std::cout << "KL divergence = " << crf.klDivergence(Q) << std::endl;
	for (int it = 0; it < 5; ++it)
	{
		crf.stepInference(Q, t1, t2);
		std::cout << "KL divergence = " << crf.klDivergence(Q) << std::endl;
	}
#endif

	std::cout << "Start infering..." << std::endl;
	//const VectorXs map(crf.currentMap(Q));
	const VectorXs map(crf.map(5));
	std::cout << "End infering..." << std::endl;

	// Store the result.
	std::unique_ptr<unsigned char> res(colorize(map, img.cols, img.rows));

	const cv::Mat rgb(img.rows, img.cols, img.type(), res.get());
	cv::imshow("DenseCRF - Image", img);
	cv::imshow("DenseCRF - Annotation", anno);
	cv::imshow("DenseCRF - Result", rgb);

	//cv::imwrite("./data/probabilistic_graphical_model/densecrf/inf_result.jpg", rgb);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

// REF [file] >> ${DENSECRF_HOME}/examples/dense_learning.cpp
void learning_example()
{
#if 1
	const std::string img_filename("./data/probabilistic_graphical_model/im1.ppm");
	const std::string anno_filename("./data/probabilistic_graphical_model/anno1.ppm");
#elif 0
	const std::string img_filename("./data/probabilistic_graphical_model/im2.ppm");
	const std::string anno_filename("./data/probabilistic_graphical_model/anno2.ppm");
#elif 0
	const std::string img_filename("./data/probabilistic_graphical_model/im3.ppm");
	const std::string anno_filename("./data/probabilistic_graphical_model/anno3.ppm");
#else
	const std::string img_filename("D:/dataset/phenotyping/RDA/20160406_trimmed_plant/adaptor1/side_0.png");
	const std::string anno_filename("D:/dataset/phenotyping/RDA/20160406_trimmed_plant/adaptor1/side_0.png.init_fg.png");
#endif

	// Number of labels.
	const int NUM_LABELS = 4;

	// Load the color image and some crude annotations (which are used in a simple classifier).
	const cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Failed to load image: " << img_filename << std::endl;
		return;
	}
	const cv::Mat anno(cv::imread(anno_filename, cv::IMREAD_COLOR));
	if (anno.empty())
	{
		std::cerr << "Failed to load annotations: " << anno_filename << std::endl;
		return;
	}
	if (img.rows != anno.rows || img.cols != anno.cols)
	{
		std::cerr << "Annotation size doesn't match image!" << std::endl;
		return;
	}

	// Get the labeling.
	const int N = anno.rows * anno.cols;
	const VectorXs labeling(getLabeling(anno.data, N, NUM_LABELS));

	std::cout << "Labeling:" << labeling.size() << " : " << labeling.minCoeff() << ", " << labeling.maxCoeff() << std::endl;

	// Get the logistic features (unary term).
	// Here we just use the color as a feature.
	MatrixXf logistic_feature(4, N), logistic_transform(NUM_LABELS, 4);
	logistic_feature.fill(1.0f);
	for (int r = 0, idx = 0; r < img.rows; ++r)
		for (int c = 0; c < img.cols; ++c, ++idx)
		{
			const cv::Vec3b bgr(img.at<cv::Vec3b>(r, c));
			logistic_feature(0, idx) = bgr[2] / 255.;
			logistic_feature(1, idx) = bgr[1] / 255.;
			logistic_feature(2, idx) = bgr[0] / 255.;
		}

	for (int j = 0; j < logistic_transform.cols(); ++j)
		for (int i = 0; i < logistic_transform.rows(); ++i)
			//--S [] 2017/05/18: Sang-Wook Lee.
			//logistic_transform(i, j) = 0.01 * (1 - 2. * random() / RAND_MAX);
			logistic_transform(i, j) = 0.01 * (1 - 2. * std::rand() / RAND_MAX);
			//--E [] 2017/05/18: Sang-Wook Lee.

	// Setup the CRF model.
	DenseCRF2D crf(img.cols, img.rows, NUM_LABELS);
	// Add a logistic unary term.
	crf.setUnaryEnergy(logistic_transform, logistic_feature);
	// Add simple pairwise potts terms.
	crf.addPairwiseGaussian(3, 3, new PottsCompatibility(1));
	// Add a longer range label compatibility term.
	crf.addPairwiseBilateral(80, 80, 13, 13, 13, img.data, new MatrixCompatibility(MatrixXf::Identity(NUM_LABELS, NUM_LABELS)));

	// Choose your loss function.
	//LogLikelihood objective(labeling, 0.01);  // Log likelihood loss.
	//Hamming objective(labeling, 0.0);  // Global accuracy.
	//Hamming objective(labeling, 1.0);  // Class average accuracy.
	//Hamming objective(labeling, 0.2);  // Hamming loss close to intersection over union.
	IntersectionOverUnion objective(labeling);  // Intersection over union accuracy.

	const int NIT = 5;
	const bool verbose = true;

	MatrixXf learning_params(3, 3);
	// Optimize the CRF in 3 phases:
	//  - First unary only.
	//  - Unary and pairwise.
	//  - Full CRF.
	learning_params << 1, 0, 0,
		1, 1, 0,
		1, 1, 1;

	for (int i = 0; i < learning_params.rows(); ++i)
	{
		// Setup the energy.
		CRFEnergy energy(crf, objective, NIT, learning_params(i, 0), learning_params(i, 1), learning_params(i, 2));
		energy.setL2Norm(1e-3);

		// Minimize the energy.
		VectorXf p = minimizeLBFGS(energy, 2, true);

		// Save the values.
		int id = 0;
		if (learning_params(i, 0))
		{
			crf.setUnaryParameters(p.segment(id, crf.unaryParameters().rows()));
			id += crf.unaryParameters().rows();
		}
		if (learning_params(i, 1))
		{
			crf.setLabelCompatibilityParameters(p.segment(id, crf.labelCompatibilityParameters().rows()));
			id += crf.labelCompatibilityParameters().rows();
		}
		if (learning_params(i, 2))
			crf.setKernelParameters(p.segment(id, crf.kernelParameters().rows()));
	}
	// Return the parameters.
	std::cout << "Unary parameters: " << crf.unaryParameters().transpose() << std::endl;
	std::cout << "Pairwise parameters: " << crf.labelCompatibilityParameters().transpose() << std::endl;
	std::cout << "Kernel parameters: " << crf.kernelParameters().transpose() << std::endl;

	// Do MAP inference.
	std::cout << "Start infering..." << std::endl;
	const VectorXs map = crf.map(NIT);
	std::cout << "End infering..." << std::endl;

	// Store the result.
	std::unique_ptr<unsigned char> res(colorize(map, img.cols, img.rows));

	const cv::Mat rgb(img.rows, img.cols, img.type(), res.get());
	cv::imshow("DenseCRF - Image", img);
	cv::imshow("DenseCRF - Annotation", anno);
	cv::imshow("DenseCRF - Result", rgb);

	//cv::imwrite("./data/probabilistic_graphical_model/densecrf/learn_result.jpg", rgb);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_densecrf {

}  // namespace my_densecrf

int densecrf_main(int argc, char *argv[])
{
	local::inference_example();
	local::learning_example();

	return 0;
}
