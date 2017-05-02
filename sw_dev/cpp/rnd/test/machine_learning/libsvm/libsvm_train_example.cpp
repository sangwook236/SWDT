#include "../libsvm_lib/svm.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cerrno>


namespace {
namespace local {

void print_null(const char *s)
{
}

void exit_input_error(int line_num)
{
	std::cout << "Wrong input format at line " << line_num << std::endl;
}

char * readline(FILE *input, char *line, int max_line_len)
{
	if (nullptr == fgets(line, max_line_len, input))
		return nullptr;

	int len;
	while (nullptr == strrchr(line, '\n'))
	{
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (nullptr == fgets(line + len, max_line_len - len, input))
			break;
	}

	return line;
}

// Read in a problem (in SVM-light format).
// SVM-light format: <target (label)> <feature id>:<value> ... <feature id>:<value> # <info> 
bool read_problem(const char *filename, struct svm_problem &prob, struct svm_parameter &param, struct svm_node *&x_space)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename, "r");
	char *endptr;
	char *idx, *val, *label;

	if (nullptr == fp)
	{
		std::cout << "Can't open input file " << filename << std::endl;
		return false;
	}

	prob.l = 0;  // Sample count.
	// If there is no missing feature, elements = sample count * (feature dimension + 1).
	//	Each instance has an additional feature with -1 as index. So, used feature dimension = actual feature dimension + 1.
	//	So features have variable length.
	// If there is missing features, elements = actual feature count + sample count.
	elements = 0;

	int max_line_len = 1024;
	char *line = (char *)malloc(sizeof(char) * max_line_len);
	while (readline(fp, line, max_line_len) != nullptr)
	{
		char *p = strtok(line, " \t");  // Label.

		// Features.
		while (true)
		{
			p = strtok(nullptr, " \t");
			if (nullptr == p || '\n' == *p)  // Check '\n' as ' ' may be after the last feature.
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = (double *)malloc(sizeof(double) * prob.l);
	prob.x = (struct svm_node **)malloc(sizeof(struct svm_node *) * prob.l);
	x_space = (struct svm_node *)malloc(sizeof(struct svm_node) * elements);

	max_index = 0;
	j = 0;
	for (i = 0; i < prob.l; ++i)
	{
		inst_max_index = -1;  // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0.
		readline(fp, line, max_line_len);
		prob.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if (nullptr == label)  // Empty line.
		{
			exit_input_error(i + 1);
			return false;
		}

		prob.y[i] = strtod(label, &endptr);
		if (endptr == label || '\0' != *endptr)
		{
			exit_input_error(i + 1);
			return false;
		}

		while (true)
		{
			idx = strtok(nullptr, ":");
			val = strtok(nullptr, " \t");

			if (nullptr == val)
				break;

			errno = 0;
			x_space[j].index = (int)strtol(idx, &endptr,10);
			if (endptr == idx || 0 != errno || '\0' != *endptr || x_space[j].index <= inst_max_index)
			{
				exit_input_error(i + 1);
				return false;
			}
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if (endptr == val || 0 != errno || ('\0' != *endptr && !isspace(*endptr)))
			{
				exit_input_error(i + 1);
				return false;
			}

			++j;
		}

		if (inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if (param.gamma == 0 && max_index > 0)
		param.gamma = 1.0 / max_index;

	if (PRECOMPUTED == param.kernel_type)
		for (i = 0; i < prob.l; ++i)
		{
			if (0 != prob.x[i][0].index)
			{
				std::cout << "Wrong input format: first column must be 0:sample_serial_number." << std::endl;
				return false;
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				std::cout << "Wrong input format: sample_serial_number out of range." << std::endl;
				return false;
			}
		}

	free(line);

	fclose(fp);
	return true;
}

void do_cross_validation(const struct svm_problem &prob, const struct svm_parameter &param, const int nr_fold)
{
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = (double *)malloc(sizeof(double) * prob.l);

	svm_cross_validation(&prob, &param, nr_fold, target);
	if (param.svm_type == EPSILON_SVR || param.svm_type == NU_SVR)
	{
		for (int i = 0; i < prob.l; ++i)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v - y) * (v - y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		std::cout << "Cross Validation Mean squared error = " << (total_error / prob.l) << std::endl;
		std::cout << "Cross Validation Squared correlation coefficient = " << ((prob.l*sumvy - sumv*sumy) * (prob.l*sumvy - sumv*sumy)) / ((prob.l*sumvv - sumv*sumv) * (prob.l*sumyy - sumy*sumy)) << std::endl;
	}
	else
	{
		for (int i = 0; i < prob.l; ++i)
			if (target[i] == prob.y[i])
				++total_correct;
		std::cout << "Cross Validation Accuracy = " << (100.0 * total_correct / prob.l) << '%' << std::endl;
	}

	free(target);
}

}  // namespace local
}  // unnamed namespace

namespace my_libsvm {

// REF [file] >> ${LIBSVM_HOME}/svm-train.c
void train_example()
{
	struct svm_parameter param;
	param.svm_type = C_SVC;  // C-SVC. Multi-class classification.
	//param.svm_type = NU_SVC;  // nu-SVC. Multi-class classification.
	//param.svm_type = ONE_CLASS;  // One-class SVM.
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

	param.degree = 3;  // Set degree in kernel function (default 3).
	param.gamma = 0;  // Set gamma in kernel function (default 1/num_features).
	param.coef0 = 0;  // Set coef0 in kernel function (default 0).
	param.C = 1.0;  // Set the parameter C of C-SVC, epsilon-SVR, nu-SVR, SVDD, and R2q (default 1, except 2/num_instances for SVDD).
	param.nu = 0.5;  // Set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5).
	param.p = 0.1;  // Set the epsilon in loss function of epsilon-SVR (default 0.1).
	param.cache_size = 100;  // Set cache memory size in MB (default 100).
	param.eps = 1e-3;  // Set tolerance of termination criterion (default 0.001).
	param.shrinking = 1;  // Whether to use the shrinking heuristics, 0 or 1 (default 1).
	param.probability = 0;  // Whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0).
	param.weight = nullptr;  // Set the parameter C of class i to weight*C, for C-SVC (default 1).
	param.weight_label = nullptr;  // For C-SVC.
	param.nr_weight = 0;  // For C-SVC.
	const bool cross_validation = false;
	const int nr_fold = 10;

	svm_set_print_string_function(local::print_null);

	const std::string input_file_name("./data/machine_learning/svm/heart_scale");
	const std::string model_file_name("./data/machine_learning/svm/heart_scale.model");

	struct svm_problem prob;
	struct svm_node *x_space = nullptr;
	local::read_problem(input_file_name.c_str(), prob, param, x_space);

	const char *error_msg = svm_check_parameter(&prob, &param);
	if (error_msg)
	{
		std::cout << "ERROR: " << error_msg << std::endl;
		return;
	}

	if (cross_validation)
	{
		local::do_cross_validation(prob, param, nr_fold);
	}
	else
	{
		struct svm_model *model = svm_train(&prob, &param);
		if (svm_save_model(model_file_name.c_str(), model))
		{
			std::cout << "Can't save model to file " << model_file_name << std::endl;
			return;
		}
		svm_free_and_destroy_model(&model);
	}

	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
}

}  // namespace my_libsvm
