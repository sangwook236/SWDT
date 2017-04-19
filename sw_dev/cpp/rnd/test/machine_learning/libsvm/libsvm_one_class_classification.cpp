#include "../libsvm_lib/svm.h"
#include <memory>
#include <ctime>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/timer/timer.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>


namespace {
namespace local {

void print_null(const char *s)
{
}

void exit_input_error(int line_num)
{
	std::cout << "Wrong input format at line " << line_num << std::endl;
}

// REF [site] >> https://stats.stackexchange.com/questions/61036/libsvm-one-class-svm-how-to-consider-all-data-to-be-in-class
bool create_problem(const int numSample, const int dimFeature, svm_problem &problem, svm_parameter &param, svm_node *x_space)
{
	problem.l = numSample;

	{
		typedef boost::minstd_rand base_generator_type;
		//typedef boost::mt19937 base_generator_type;
		base_generator_type baseGenerator(static_cast<unsigned int>(std::time(NULL)));

		typedef boost::normal_distribution<> distribution_type;
		typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

		const double mean = 0.0;
		const double sigma = 1.0;
		generator_type normal_gen(baseGenerator, distribution_type(mean, sigma));
		int idx = 0;
		for (int i = 0; i < problem.l; ++i)
		{
			problem.x[i] = x_space + idx;  // Feature.
			problem.y[i] = 1.0;  // Target (label).
			for (int f = 1; f <= dimFeature; ++f, ++idx)
			{
				x_space[idx].index = f;  // One-based index. But it can have index 0 in using PRECOMPUTED.
				x_space[idx].value = normal_gen();
			}
			x_space[idx].index = -1;
			x_space[idx].value = 0.0;
			++idx;
		}
	}

	if (0 == param.gamma && numSample > 0)
		param.gamma = 1.0 / numSample;

	/*
	if (PRECOMPUTED == param.kernel_type)
		for (int i = 0; i < problem.l; ++i)
		{
			if (0 != problem.x[i][0].index)
			{
				std::cout << "Wrong input format: first column must be 0:sample_serial_number." << std::endl;
				return false;
			}
			if ((int)problem.x[i][0].value <= 0 || (int)problem.x[i][0].value > numSample)
			{
				std::cout << "Wrong input format: sample_serial_number out of range." << std::endl;
				return false;
			}
		}
	*/

	return true;
}

bool writeToFile(const std::string &data_file_name, const int numInstances, const svm_node **dataNode, const std::vector<int> &svIndices, const std::vector<int> &inliers)
{
	std::ofstream stream(data_file_name.c_str(), std::ios::trunc);
	if (stream)
	{
		const svm_node **node = dataNode;
		for (int i = 1; i <= numInstances; ++i, ++node)
		{
			const svm_node *feat = *node;
			stream << i << ' ';
			while (-1 != feat->index)
			{
				stream << feat->value << ' ';
				++feat;
			}
			stream << std::endl;
		}
		stream << std::endl;

		for (const auto &sv : svIndices)
		{
			const svm_node *feat = dataNode[sv - 1];
			stream << sv << ' ';
			while (-1 != feat->index)
			{
				stream << feat->value << ' ';
				++feat;
			}
			stream << std::endl;
		}
		stream << std::endl;

		for (const auto &inlier : inliers)
		{
			const svm_node *feat = dataNode[inlier - 1];
			stream << inlier << ' ';
			while (-1 != feat->index)
			{
				stream << feat->value << ' ';
				++feat;
			}
			stream << std::endl;
		}

		return true;
	}
	else
	{
		std::cerr << "File not found: " << data_file_name << std::endl;
		return false;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_libsvm {

void simple_svdd()
{
	const int NUM_INSTANCES = 1000;  // Sample size.
	const int DIM_FEATURES = 2;  // Feature dimension.

	svm_parameter param;
	//param.svm_type = C_SVC;  // C-SVC. Multi-class classification.
	//param.svm_type = NU_SVC;  // nu-SVC. Multi-class classification.
	//param.svm_type = ONE_CLASS;  // One-class SVM.
	//param.svm_type = EPSILON_SVR;  // epsilon-SVR. Regression.
	//param.svm_type = NU_SVR;  // nu-SVR. Regression.
	param.svm_type = SVDD;  // C should be between 1/num_instances and 1.
	//param.svm_type = R2;  // R^2: L1SVM.
	//param.svm_type = R2q;  // R^2: L2SVM.

	//param.kernel_type = LINEAR;  // Linear: u'*v.
	//param.kernel_type = POLY;  // Polynomial: (gamma*u'*v + coef0)^degree.
	param.kernel_type = RBF;  // Radial basis function: exp(-gamma*|u-v|^2).
	//param.kernel_type = SIGMOID;  // Sigmoid: tanh(gamma*u'*v + coef0).
	//param.kernel_type = PRECOMPUTED;  // Precomputed kernel (kernel values in training_set_file).

	// For SVDD:
	//	If param.C = 1.0 / NUM_INSTANCES, SV: 100%, inlier: 0%.
	//	If param.C = 2.0 / NUM_INSTANCES, SV: ~50%, inlier: ~50%.
	//	If param.C = 4.0 / NUM_INSTANCES, SV: ~25%, inlier: ~75%.
	//	If param.C = 8.0 / NUM_INSTANCES, SV: ~14%, inlier: ~78%.
	//	If param.C = 16.0 / NUM_INSTANCES, SV: ~10%, inlier: ~84%.

	param.degree = 3;  // Set degree in kernel function (default 3).
	param.gamma = 1.0 / DIM_FEATURES;  // Set gamma in kernel function (default 1/num_features).
	param.coef0 = 0;  // Set coef0 in kernel function (default 0).
	param.C = 4.0 / NUM_INSTANCES;  // Set the parameter C of C-SVC, epsilon-SVR, nu-SVR, SVDD, and R2q (default 1, except 2/num_instances for SVDD).
	param.nu = 0.5;  // Set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5).
	param.p = 0.1;  // Set the epsilon in loss function of epsilon-SVR (default 0.1).
	param.cache_size = 100;  // Set cache memory size in MB (default 100).
	param.eps = 1e-3;  // Set tolerance of termination criterion (default 0.001).
	param.shrinking = 1;  // Whether to use the shrinking heuristics, 0 or 1 (default 1).
	param.probability = 0;  // Whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0).
	param.weight = nullptr;  // Set the parameter C of class i to weight*C, for C-SVC (default 1).
	param.weight_label = nullptr;  // For C-SVC.
	param.nr_weight = 0;  // For C-SVC.

	svm_set_print_string_function(local::print_null);

	const std::string model_file_name("./data/machine_learning/svm/simple_svdd.model");
	const std::string data_file_name("./data/machine_learning/svm/simple_svdd.dat");

	// Every instance has an additional feature with -1 as index.
	// Features have variable length if there is any missing feature.
#if 1
	std::unique_ptr<svm_node *> problemX(new svm_node * [NUM_INSTANCES]);
	std::unique_ptr<double> problemY(new double [NUM_INSTANCES]);

	svm_problem problem;
	problem.l = NUM_INSTANCES;
	problem.x = problemX.get();
	problem.y = problemY.get();
	std::unique_ptr<svm_node> x_space(new svm_node [problem.l * (DIM_FEATURES + 1)]);

	local::create_problem(NUM_INSTANCES, DIM_FEATURES, problem, param, x_space.get());
#elif 0
	svm_problem problem;
	problem.l = NUM_INSTANCES;
	problem.y = new double[problem.l];
	problem.x = new svm_node *[problem.l];
	svm_node *x_space = new svm_node[problem.l * (DIM_FEATURES + 1)];

	local::create_problem(NUM_INSTANCES, DIM_FEATURES, problem, param, x_space);
#else
	svm_problem problem;
	problem.l = NUM_INSTANCES;
	problem.y = (double *)malloc(sizeof(double) * problem.l);
	problem.x = (svm_node **)malloc(sizeof(svm_node *) * problem.l);
	svm_node *x_space = (svm_node *)malloc(sizeof(svm_node) * problem.l * (DIM_FEATURES + 1));

	local::create_problem(NUM_INSTANCES, DIM_FEATURES, problem, param, x_space);
#endif

	const char *error_msg = svm_check_parameter(&problem, &param);
	if (error_msg)
	{
		std::cout << "ERROR: " << error_msg << std::endl;
		return;
	}

	// Train.
	svm_model *model = nullptr;
	{
		std::cout << "Start training..." << std::endl;
		{
			boost::timer::auto_cpu_timer timer;
			model = svm_train(&problem, &param);
		}
		std::cout << "End training..." << std::endl;

		// Save model.
		//if (svm_save_model(model_file_name.c_str(), model))
		//{
		//	std::cout << "Can't save model to file " << model_file_name << std::endl;
		//	return;
		//}
	}

	// Predict.
	if (nullptr != model)
	{
		const int svmType = svm_get_svm_type(model);
		const int numClasses = svm_get_nr_class(model);

		const bool predictProbability = false;
		if (predictProbability)
		{
			if (0 == svm_check_probability_model(model))
			{
				std::cout << "Model does not support probabiliy estimates." << std::endl;
				return;
			}
		}
		else
		{
			if (0 != svm_check_probability_model(model))
				std::cout << "Model supports probability estimates, but disabled in prediction." << std::endl;
		}

#if 0
		if (predictProbability && (C_SVC == svmType || NU_SVC == svmType))
		{
			std::vector<svm_node> x(DIM_FEATURES);
			std::vector<double> probilityEstimates(numClasses, 0.0);
			const double predictedLabel = svm_predict_probability(model, &x[0], &probilityEstimates[0]);

			std::cout << "Predicted label = " << predictedLabel << std::endl;
			std::cout << "Estimated class probability = ";
			for (int j = 0; j < numClasses; ++j)
				std::cout << probilityEstimates[j] << ", ";
			std::cout << std::endl;
		}
		if (predictProbability && (NU_SVR == svmType || EPSILON_SVR == svmType))
			std::cout << "Prob. model for test data: target value = predicted value + z:" << std::endl
				<< "\tz: Laplace distribution e^(-|z|/sigma)/(2sigma), sigma = " << svm_get_svr_probability(model) << std::endl;
		else
		{
			std::vector<svm_node> x(DIM_FEATURES);
			const double predictedValue = svm_predict(model, &x[0]);

			std::cout << "Predicted value = " << predictedValue << std::endl;
		}
#else
		std::vector<int> inliers;  // One-based index.
		{
			svm_node **node = problem.x;
			inliers.reserve(NUM_INSTANCES);
			std::cout << "Start predicting..." << std::endl;
			for (int i = 1; i <= NUM_INSTANCES; ++i, ++node)
			{
				const double predictedLabel = svm_predict(model, *node);
				if (predictedLabel > 0.0) inliers.push_back(i);
			}
			std::cout << "End predicting..." << std::endl;
		}

		//
		const int numSV = svm_get_nr_sv(model);
		std::vector<int> svIndices(numSV, -1);  // One-based index.
		svm_get_sv_indices(model, &svIndices[0]);

		std::cout << "#inliers = " << inliers.size() << std::endl;
		std::cout << "#SV = " << numSV << std::endl;

		// Write to a file.
		local::writeToFile(data_file_name, NUM_INSTANCES, (const svm_node **)problem.x, svIndices, inliers);
#endif
	}

	// Clean-up.
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);
}

void simple_one_class_svm()
{
	const int NUM_INSTANCES = 1000;  // Sample size.
	const int DIM_FEATURES = 2;  // Feature dimension.

	svm_parameter param;
	//param.svm_type = C_SVC;  // C-SVC. Multi-class classification.
	//param.svm_type = NU_SVC;  // nu-SVC. Multi-class classification.
	param.svm_type = ONE_CLASS;  // One-class SVM.
	//param.svm_type = EPSILON_SVR;  // epsilon-SVR. Regression.
	//param.svm_type = NU_SVR;  // nu-SVR. Regression.
	//param.svm_type = SVDD;  // C should be between 1/num_instances and 1.
	//param.svm_type = R2;  // R^2: L1SVM.
	//param.svm_type = R2q;  // R^2: L2SVM.

	//param.kernel_type = LINEAR;  // Linear: u'*v.
	//param.kernel_type = POLY;  // Polynomial: (gamma*u'*v + coef0)^degree.
	param.kernel_type = RBF;  // Radial basis function: exp(-gamma*|u-v|^2).
	//param.kernel_type = SIGMOID;  // Sigmoid: tanh(gamma*u'*v + coef0).
	//param.kernel_type = PRECOMPUTED;  // Precomputed kernel (kernel values in training_set_file).

	// For one-class SVM:
	//	If param.nu = 1.0, SV: 100%, inlier: 0%.
	//	If param.nu = 0.5, SV: ~50%, inlier: ~50%.
	//	If param.nu = 0.2, SV: ~20%, inlier: ~80%.
	//	If param.nu = 0.1, SV: ~10%, inlier: ~90%.

	param.degree = 3;  // Set degree in kernel function (default 3).
	param.gamma = 1.0 / DIM_FEATURES;  // Set gamma in kernel function (default 1/num_features).
	param.coef0 = 0;  // Set coef0 in kernel function (default 0).
	param.C = 1;  // Set the parameter C of C-SVC, epsilon-SVR, nu-SVR, SVDD, and R2q (default 1, except 2/num_instances for SVDD).
	param.nu = 0.2;  // Set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5).
	param.p = 0.1;  // Set the epsilon in loss function of epsilon-SVR (default 0.1).
	param.cache_size = 100;  // Set cache memory size in MB (default 100).
	param.eps = 1e-3;  // Set tolerance of termination criterion (default 0.001).
	param.shrinking = 1;  // Whether to use the shrinking heuristics, 0 or 1 (default 1).
	param.probability = 0;  // Whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0).
	param.weight = nullptr;  // Set the parameter C of class i to weight*C, for C-SVC (default 1).
	param.weight_label = nullptr;  // For C-SVC.
	param.nr_weight = 0;  // For C-SVC.

	svm_set_print_string_function(local::print_null);

	const std::string model_file_name("./data/machine_learning/svm/simple_svdd.model");
	const std::string data_file_name("./data/machine_learning/svm/simple_one_class_svm.dat");

	// Every instance has an additional feature with -1 as index.
	// Features have variable length if there is any missing feature.
#if 1
	std::unique_ptr<svm_node *> problemX(new svm_node * [NUM_INSTANCES]);
	std::unique_ptr<double> problemY(new double [NUM_INSTANCES]);

	svm_problem problem;
	problem.l = NUM_INSTANCES;
	problem.x = problemX.get();
	problem.y = problemY.get();
	std::unique_ptr<svm_node> x_space(new svm_node [problem.l * (DIM_FEATURES + 1)]);

	local::create_problem(NUM_INSTANCES, DIM_FEATURES, problem, param, x_space.get());
#elif 0
	svm_problem problem;
	problem.l = NUM_INSTANCES;
	problem.y = new double[problem.l];
	problem.x = new svm_node *[problem.l];
	svm_node *x_space = new svm_node[problem.l * (DIM_FEATURES + 1)];

	local::create_problem(NUM_INSTANCES, DIM_FEATURES, problem, param, x_space);
#else
	svm_problem problem;
	problem.l = NUM_INSTANCES;
	problem.y = (double *)malloc(sizeof(double) * problem.l);
	problem.x = (svm_node **)malloc(sizeof(svm_node *) * problem.l);
	svm_node *x_space = (svm_node *)malloc(sizeof(svm_node) * problem.l * (DIM_FEATURES + 1));

	local::create_problem(NUM_INSTANCES, DIM_FEATURES, problem, param, x_space);
#endif

	const char *error_msg = svm_check_parameter(&problem, &param);
	if (error_msg)
	{
		std::cout << "ERROR: " << error_msg << std::endl;
		return;
	}

	// Train.
	svm_model *model = nullptr;
	{
		std::cout << "Start training..." << std::endl;
		{
			boost::timer::auto_cpu_timer timer;
			model = svm_train(&problem, &param);
		}
		std::cout << "End training..." << std::endl;

		// Save model.
		//if (svm_save_model(model_file_name.c_str(), model))
		//{
		//	std::cout << "Can't save model to file " << model_file_name << std::endl;
		//	return;
		//}
	}

	// Predict.
	if (nullptr != model)
	{
		const int svmType = svm_get_svm_type(model);
		const int numClasses = svm_get_nr_class(model);

		const bool predictProbability = false;
		if (predictProbability)
		{
			if (0 == svm_check_probability_model(model))
			{
				std::cout << "Model does not support probabiliy estimates." << std::endl;
				return;
			}
		}
		else
		{
			if (0 != svm_check_probability_model(model))
				std::cout << "Model supports probability estimates, but disabled in prediction." << std::endl;
		}

#if 0
		if (predictProbability && (C_SVC == svmType || NU_SVC == svmType))
		{
			std::vector<svm_node> x(DIM_FEATURES);
			std::vector<double> probilityEstimates(numClasses, 0.0);
			const double predictedLabel = svm_predict_probability(model, &x[0], &probilityEstimates[0]);

			std::cout << "Predicted label = " << predictedLabel << std::endl;
			std::cout << "Estimated class probability = ";
			for (int j = 0; j < numClasses; ++j)
				std::cout << probilityEstimates[j] << ", ";
			std::cout << std::endl;
		}
		if (predictProbability && (NU_SVR == svmType || EPSILON_SVR == svmType))
			std::cout << "Prob. model for test data: target value = predicted value + z:" << std::endl
				<< "\tz: Laplace distribution e^(-|z|/sigma)/(2sigma), sigma = " << svm_get_svr_probability(model) << std::endl;
		else
		{
			std::vector<svm_node> x(DIM_FEATURES);
			const double predictedValue = svm_predict(model, &x[0]);

			std::cout << "Predicted value = " << predictedValue << std::endl;
		}
#else
		std::vector<int> inliers;  // One-based index.
		{
			svm_node **node = problem.x;
			inliers.reserve(NUM_INSTANCES);
			std::cout << "Start predicting..." << std::endl;
			for (int i = 1; i <= NUM_INSTANCES; ++i, ++node)
			{
				const double predictedLabel = svm_predict(model, *node);
				if (predictedLabel > 0.0) inliers.push_back(i);
			}
			std::cout << "End predicting..." << std::endl;
		}

		//
		const int numSV = svm_get_nr_sv(model);
		std::vector<int> svIndices(numSV, -1);  // One-based index.
		svm_get_sv_indices(model, &svIndices[0]);

		std::cout << "#inliers = " << inliers.size() << std::endl;
		std::cout << "#SV = " << numSV << std::endl;

		// Write to a file.
		local::writeToFile(data_file_name, NUM_INSTANCES, (const svm_node **)problem.x, svIndices, inliers);
#endif
	}

	// Clean-up.
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);
}

}  // namespace my_libsvm
