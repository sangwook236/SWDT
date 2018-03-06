#include <densecrf/densecrf.h>
#include <densecrf/optimization.h>
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <cmath>
#include <memory>


namespace {
namespace local {

// REF [paper] >> "Parameter Learning and Convergent Inference for Dense Random Fields", ICML 2013.
// REF [paper] >> "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials", NIPS 2011.
// REF [paper] >> "Filter-Based Mean-Field Inference for Random Fields with Higher-Order Terms and Product Label-Spaces", IJCV 2014.

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
		for (i = 0; i < nColors && c != colors[i]; ++i);
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
	const float u_energy = -(float)std::log(1.0f / NUM_LABELS);
	const float n_energy = -(float)std::log((1.0f - GT_PROB) / (NUM_LABELS - 1.0f));
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
	: crf_(crf), objective_(objective), NIT_(NIT), unary_(unary), pairwise_(pairwise), kernel_(kernel), l2_norm_(0.0f)
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
			r += 0.5 * l2_norm_ * (x.dot(x));
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
	const std::string img_filename("../bin/data/probabilistic_graphical_model/im1.ppm");
	const std::string anno_filename("../bin/data/probabilistic_graphical_model/anno1.ppm");
#elif 0
	const std::string img_filename("../bin/data/probabilistic_graphical_model/im2.ppm");
	const std::string anno_filename("../bin/data/probabilistic_graphical_model/anno2.ppm");
#elif 0
	const std::string img_filename("../bin/data/probabilistic_graphical_model/im3.ppm");
	const std::string anno_filename("../bin/data/probabilistic_graphical_model/anno3.ppm");
#endif

	// Load the color image and some crude annotations (which are used in a simple classifier).
	const cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Failed to load image: " << img_filename << std::endl;
		return;
	}
	cv::Mat anno(cv::imread(anno_filename, cv::IMREAD_COLOR));
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

#if 0
	// NOTE [info] {important} >> Here 0 means the unknown label. So the background label has to be changed from 0 to non-zero.
	{
		cv::Mat anno_bg;
		cv::cvtColor(anno, anno_bg, cv::COLOR_BGR2GRAY);
		anno_bg = cv::Scalar::all(255) - anno_bg;

		const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		cv::morphologyEx(anno_bg, anno_bg, cv::MORPH_ERODE, selement, cv::Point(-1, -1), 5);

		anno.setTo(cv::Scalar::all(128), 0 != anno_bg);  // BG.
	}
#endif

	// Number of labels.
	const int NUM_LABELS = 21;

	// Setup the CRF model.
	DenseCRF2D crf(img.cols, img.rows, NUM_LABELS);
	{
		// Put your own unary classifier here.
		// Certainty that the groundtruth is correct.
		const float GT_PROB = 0.5f;
		// Simple classifier that is 50% certain that the annotation is correct.
		// FIXME [improve] >> Too simple unary computed by GT_PROB is used here.
		const MatrixXf unary(computeUnary(getLabeling(anno.data, anno.rows * anno.cols, NUM_LABELS), NUM_LABELS, GT_PROB));

		// Specify the unary potential as an array of size W*H*(#classes).
		//	Packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
		crf.setUnaryEnergy(unary);
	}
	{
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
		//crf.addPairwiseBilateral(80, 80, 13, 13, 13, img.data, new MatrixCompatibility(MatrixXf::Identity(NUM_LABELS, NUM_LABELS)));
	}

	// Do MAP inference.
	std::cout << "Start infering..." << std::endl;
	const int num_iterations = 5;
	boost::timer::cpu_timer timer;
#if 0
	MatrixXf Q = crf.startInference(), t1, t2;
	std::cout << "KL divergence = " << crf.klDivergence(Q) << std::endl;
	for (int it = 0; it < num_iterations; ++it)
	{
		crf.stepInference(Q, t1, t2);
		std::cout << "KL divergence = " << crf.klDivergence(Q) << std::endl;
	}
	const VectorXs map(crf.currentMap(Q));
#else
	const VectorXs map(crf.map(num_iterations));
#endif
	std::cout << timer.format() << std::endl;
	timer.stop();
	std::cout << "End infering..." << std::endl;

	// Store the result.
	std::unique_ptr<unsigned char> res(colorize(map, img.cols, img.rows));

	const cv::Mat rgb(img.rows, img.cols, img.type(), res.get());
	cv::imshow("DenseCRF - Image", img);
	cv::imshow("DenseCRF - Annotation", anno);
	cv::imshow("DenseCRF - Result", rgb);

	//cv::imwrite("../bin/data/probabilistic_graphical_model/densecrf/inf_result.jpg", rgb);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

// REF [file] >> ${DENSECRF_HOME}/examples/dense_learning.cpp
void learning_example()
{
#if 1
	const std::string img_filename("../bin/data/probabilistic_graphical_model/im1.ppm");
	const std::string anno_filename("../bin/data/probabilistic_graphical_model/anno1.ppm");
#elif 0
	const std::string img_filename("../bin/data/probabilistic_graphical_model/im2.ppm");
	const std::string anno_filename("../bin/data/probabilistic_graphical_model/anno2.ppm");
#elif 0
	const std::string img_filename("../bin/data/probabilistic_graphical_model/im3.ppm");
	const std::string anno_filename("../bin/data/probabilistic_graphical_model/anno3.ppm");
#endif

	// Load the color image and some crude annotations (which are used in a simple classifier).
	const cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Failed to load image: " << img_filename << std::endl;
		return;
	}
	cv::Mat anno(cv::imread(anno_filename, cv::IMREAD_COLOR));
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

#if 0
	// NOTE [info] {important} >> Here 0 means the unknown label. So the background label has to be changed from 0 to non-zero.
	{
		cv::Mat anno_bg;
		cv::cvtColor(anno, anno_bg, cv::COLOR_BGR2GRAY);
		anno_bg = cv::Scalar::all(255) - anno_bg;

		const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		cv::morphologyEx(anno_bg, anno_bg, cv::MORPH_ERODE, selement, cv::Point(-1, -1), 5);

		anno.setTo(cv::Scalar::all(128), 0 != anno_bg);  // BG.
	}
#endif

	// Number of labels.
	const int NUM_LABELS = 4;

	// Setup the CRF model.
	DenseCRF2D crf(img.cols, img.rows, NUM_LABELS);
	{
		// Get the logistic features (unary term).
		// Here we just use the color as a feature.
		// FIXME [improve] >> Use more powerful features.
		//	- The result of a TextonBoost classifier.
		//		REF [paper] >> "Parameter Learning and Convergent Inference for Dense Random Fields", ICML 2013.
		//	- The label or features of CNN.
		MatrixXf logistic_feature(4, img.rows * img.cols), logistic_transform(NUM_LABELS, 4);
		logistic_feature.fill(1.0f);
		for (int r = 0, idx = 0; r < img.rows; ++r)
			for (int c = 0; c < img.cols; ++c, ++idx)
			{
				const cv::Vec3b bgr(img.at<cv::Vec3b>(r, c));
				logistic_feature(0, idx) = bgr[2] / 255.0f;
				logistic_feature(1, idx) = bgr[1] / 255.0f;
				logistic_feature(2, idx) = bgr[0] / 255.0f;
			}

		// Initialize logistic transform.
		for (int j = 0; j < logistic_transform.cols(); ++j)
			for (int i = 0; i < logistic_transform.rows(); ++i)
			{
				//--S [] 2017/05/18: Sang-Wook Lee.
				//logistic_transform(i, j) = 0.01f * (1.0f - 2.0f * (float)random() / (float)RAND_MAX);
				logistic_transform(i, j) = 0.01f * (1.0f - 2.0f * (float)std::rand() / (float)RAND_MAX);
				//--E [] 2017/05/18: Sang-Wook Lee.
			}

		// Add a logistic unary term.
		crf.setUnaryEnergy(logistic_transform, logistic_feature);
	}
	{
		// Add simple pairwise Potts terms.
		//	x_stddev = 3.
		//	y_stddev = 3.
		//	weight = 3.
		crf.addPairwiseGaussian(3, 3, new PottsCompatibility(3));

		// Add a longer range label compatibility term.
		//	x_stddev = 80.
		//	y_stddev = 80.
		//	r_stddev = g_stddev = b_stddev = 13.
		//	weight = 10.
		crf.addPairwiseBilateral(80, 80, 13, 13, 13, img.data, new MatrixCompatibility(MatrixXf::Identity(NUM_LABELS, NUM_LABELS)));
		//crf.addPairwiseBilateral(80, 80, 13, 13, 13, img.data, new PottsCompatibility(10));
	}

	// Get the labeling.
	const VectorXs labeling(getLabeling(anno.data, anno.rows * anno.cols, NUM_LABELS));
	std::cout << "Labeling: " << labeling.size() << " : " << labeling.minCoeff() << ", " << labeling.maxCoeff() << std::endl;

	// Choose your loss function.
	LogLikelihood objective(labeling, 0.01);  // Log likelihood loss.
	//Hamming objective(labeling, 0.0);  // Global accuracy.
	//Hamming objective(labeling, 1.0);  // Class average accuracy.
	//Hamming objective(labeling, 0.2);  // Hamming loss close to intersection over union.
	//IntersectionOverUnion objective(labeling);  // Intersection over union accuracy.

	const int num_iterations = 5;
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
		CRFEnergy energy(crf, objective, num_iterations, learning_params(i, 0), learning_params(i, 1), learning_params(i, 2));
		energy.setL2Norm(1e-3);

		// Minimize the energy.
		boost::timer::cpu_timer timer;
		const VectorXf &p = minimizeLBFGS(energy, 2, verbose);
		std::cout << timer.format() << std::endl;
		timer.stop();

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
	boost::timer::cpu_timer timer;
	const VectorXs map(crf.map(num_iterations));
	std::cout << timer.format() << std::endl;
	timer.stop();
	std::cout << "End infering..." << std::endl;

	// Store the result.
	std::unique_ptr<unsigned char> res(colorize(map, img.cols, img.rows));

	const cv::Mat rgb(img.rows, img.cols, img.type(), res.get());
	cv::imshow("DenseCRF - Image", img);
	cv::imshow("DenseCRF - Annotation", anno);
	cv::imshow("DenseCRF - Result", rgb);

	//cv::imwrite("../bin/data/probabilistic_graphical_model/densecrf/learn_result.jpg", rgb);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

void color_filtering(const cv::Mat& img, cv::Mat& foreground_mask)
{
	cv::Mat img2 = img;
	//cv::cvtColor(img, img2, CV_BGR2GRAY);  // BGR -> gray.
	//cv::cvtColor(img, img2, CV_BGR2XYZ);  // BGR -> CIE XYZ.
	//cv::cvtColor(img, img2, CV_BGR2YCrCb);  // BGR -> YCrCb JPEG.
	//cv::cvtColor(img, img2, CV_BGR2HSV);  // BGR -> HSV.
	//cv::cvtColor(img, img2, CV_BGR2HLS);  // BGR -> HLS.
	//cv::cvtColor(img, img2, CV_BGR2Lab);  // BGR -> CIE L*a*b*.
	//cv::cvtColor(img, img2, CV_BGR2Luv);  // BGR -> CIE L*u*v*.

	std::vector<cv::Mat> filtered_imgs;
	cv::split(img2, filtered_imgs);

#if 0
	std::vector<cv::Mat> filtered_imgs2(3);
	cv::equalizeHist(filtered_imgs[0], filtered_imgs2[0]);
	cv::equalizeHist(filtered_imgs[1], filtered_imgs2[1]);
	cv::equalizeHist(filtered_imgs[2], filtered_imgs2[2]);
#else
	std::vector<cv::Mat> filtered_imgs2(filtered_imgs);
#endif

	std::vector<cv::Mat> filtered_imgs3(3);
	{
		filtered_imgs3[0] = filtered_imgs2[0];
		cv::Mat mask1((filtered_imgs2[1] > 40) & (filtered_imgs2[1] < 170));
		cv::Mat mask2((filtered_imgs2[1] > filtered_imgs2[0]) & (filtered_imgs2[1] > filtered_imgs2[2]));
		//cv::Mat mask3((((float)filtered_imgs2[0] / (float)filtered_imgs2[1]) < 0.9f) & (((float)filtered_imgs2[2] / (float)filtered_imgs2[1]) < 0.9f));
		cv::Mat mask3 = cv::Mat::zeros(filtered_imgs2[1].size(), filtered_imgs2[1].type());
		for (int r = 0; r < mask3.rows; ++r)
			for (int c = 0; c < mask3.cols; ++c)
			{
				const unsigned char &green = filtered_imgs2[1].at<unsigned char>(r, c);
				if (0 == green) continue;

				const unsigned char &blue = filtered_imgs2[0].at<unsigned char>(r, c);
				const unsigned char &red = filtered_imgs2[2].at<unsigned char>(r, c);

				const float rg_ratio = (float)red / (float)green;
				const float bg_ratio = (float)blue / (float)green;
#if 1
				if (rg_ratio < 0.98f && bg_ratio < 0.98f)
#else
				const float br_ratio = 0 == red ? 1.0f : ((float)blue / (float)red);
				if (rg_ratio < 0.98f && bg_ratio < 0.98f && br_ratio < 0.95f)
#endif
					mask3.at<unsigned char>(r, c) = 1;
			}

		//filtered_imgs2[1].copyTo(filtered_imgs3[1], mask1);
		//filtered_imgs2[1].copyTo(filtered_imgs3[1], mask2);
		filtered_imgs2[1].copyTo(filtered_imgs3[1], mask3);
		//filtered_imgs2[1].copyTo(filtered_imgs3[1], mask1 & mask2);

		filtered_imgs3[1].setTo(255, filtered_imgs3[1] > 0);

		//filtered_imgs3[0] = filtered_imgs2[0];
		//filtered_imgs3[2] = filtered_imgs2[2];
	}

	foreground_mask = filtered_imgs3[1];
}

void foreground_extraction_inference_for_foreign_body()
{
#if 1
	const std::string img_filename("D:/dataset/failure_analysis/defect/xray/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/image_equalized/Bad_hamburger_164620_2136000.tif");
	const std::string anno_filename("D:/work_biz/failure_analysis_git/proto/hamburger_patty/result/fc_densenet_using_foreign_body_loader/patty_sample_std_rmsprop_lr1e-5_decay1e-9_batch4_epoch200/prediction/prediction14_1000x1024.jpg");
	//const std::string img_filename("D:/dataset/failure_analysis/defect/xray/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/patty/image_equalized/Bad_hamburger_164739_2214828.tif");
	//const std::string anno_filename("D:/work_biz/failure_analysis_git/proto/hamburger_patty/result/fc_densenet_using_foreign_body_loader/patty_sample_std_rmsprop_lr1e-5_decay1e-9_batch4_epoch200/prediction/prediction15_1000x1024.jpg");
#else
	const std::string img_filename("D:/dataset/failure_analysis/defect/xray/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/FB_trimmed/image_equalized/Bad_hamburger_164620_2136000.tif");
	const std::string anno_filename("D:/work_biz/failure_analysis_git/proto/hamburger_patty/result/fc_densenet_using_foreign_body_loader/fb_sample_std_rmsprop_lr1e-3_decay1e-7_batch4_epoch1000/prediction/prediction9_400x400.jpg");
	//const std::string img_filename("D:/dataset/failure_analysis/defect/xray/xray_201702/2016y10m03d hamburger(OK_or_NG)_processed/FB_trimmed/image_equalized/Bad_hamburger_164739_2214828.tif");
	//const std::string anno_filename("D:/work_biz/failure_analysis_git/proto/hamburger_patty/result/fc_densenet_using_foreign_body_loader/fb_sample_std_rmsprop_lr1e-3_decay1e-7_batch4_epoch1000/prediction/prediction10_400x400.jpg");
#endif

	// Load the color image and some crude annotations (which are used in a simple classifier).
	const cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Failed to load image: " << img_filename << std::endl;
		return;
	}
	cv::Mat anno(cv::imread(anno_filename, cv::IMREAD_COLOR));
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

#if 1
	// NOTE [info] {important} >> Here 0 means the unknown label. So the background label has to be changed from 0 to non-zero.
	{
		cv::Mat anno_gray;
		cv::cvtColor(anno, anno_gray, cv::COLOR_BGR2GRAY);

		cv::Mat anno_bg = cv::Scalar::all(255) - anno_gray;

		const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		cv::morphologyEx(anno_bg, anno_bg, cv::MORPH_ERODE, selement, cv::Point(-1, -1), 30);
		cv::morphologyEx(anno_gray, anno_gray, cv::MORPH_ERODE, selement, cv::Point(-1, -1), 5);

		cv::cvtColor(anno_gray, anno, cv::COLOR_GRAY2BGR);
		anno.setTo(cv::Scalar::all(128), 0 != anno_bg);  // BG.
		anno.setTo(cv::Scalar::all(255), 0 != anno_gray);  // FG.
	}
#endif

	// Number of labels.
	const int NUM_LABELS = 2;

	// Setup the CRF model.
	DenseCRF2D crf(img.cols, img.rows, NUM_LABELS);
	{
		// Put your own unary classifier here.
		// Certainty that the groundtruth is correct.
		const float GT_PROB = 0.9f;
		// Simple classifier that is 50% certain that the annotation is correct.
		// FIXME [improve] >> Too simple unary computed by GT_PROB is used here.
		const MatrixXf unary(computeUnary(getLabeling(anno.data, anno.rows * anno.cols, NUM_LABELS), NUM_LABELS, GT_PROB));

		// Specify the unary potential as an array of size W*H*(#classes).
		//	Packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
		crf.setUnaryEnergy(unary);
	}
	{
		// Add a color independent term (feature = pixel location 0..W-1, 0..H-1).
		//	x_stddev = y_stddev = 3.
		//	weight = 3.
		crf.addPairwiseGaussian(1, 1, new PottsCompatibility(4));

		// Add a color dependent term (feature = xyrgb).
		//	x_stddev = y_stddev = 60.
		//	r_stddev = g_stddev = b_stddev = 20.
		//	weight = 10.
		crf.addPairwiseBilateral(20, 20, 4, 4, 4, img.data, new PottsCompatibility(3));
		//crf.addPairwiseBilateral(80, 80, 13, 13, 13, img.data, new MatrixCompatibility(MatrixXf::Identity(NUM_LABELS, NUM_LABELS)));
	}

	// Do MAP inference.
	std::cout << "Start infering..." << std::endl;
	const int num_iterations = 5;
	boost::timer::cpu_timer timer;
#if 0
	MatrixXf Q = crf.startInference(), t1, t2;
	std::cout << "KL divergence = " << crf.klDivergence(Q) << std::endl;
	for (int it = 0; it < num_iterations; ++it)
	{
		crf.stepInference(Q, t1, t2);
		std::cout << "KL divergence = " << crf.klDivergence(Q) << std::endl;
	}
	const VectorXs map(crf.currentMap(Q));
#else
	const VectorXs map(crf.map(num_iterations));
#endif
	std::cout << timer.format() << std::endl;
	timer.stop();
	std::cout << "End infering..." << std::endl;

	// Store the result.
	std::unique_ptr<unsigned char> res(colorize(map, img.cols, img.rows));

	const cv::Mat rgb(img.rows, img.cols, img.type(), res.get());
	cv::imshow("DenseCRF - Image", img);
	cv::imshow("DenseCRF - Annotation", anno);
	cv::imshow("DenseCRF - Result", rgb);

	//cv::imwrite("../bin/data/probabilistic_graphical_model/densecrf/inf_result.jpg", rgb);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_densecrf {

}  // namespace my_densecrf

int densecrf_main(int argc, char *argv[])
{
	// Example.
	//local::inference_example();  // No parameter learning.
	local::learning_example();

	// Application: plant phenotyping.
	//	REF [directory] >> ${KIST_PROJECT_HOME}/test/dense_crf_test

	// Application: defect analysis.
	//local::foreground_extraction_inference_for_foreign_body();

	return 0;
}
