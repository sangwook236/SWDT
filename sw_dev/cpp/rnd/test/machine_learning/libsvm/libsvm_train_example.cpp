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
	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	int len;
	while (strrchr(line, '\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}

	return line;
}

// read in a problem (in svmlight format)
bool read_problem(const char *filename, struct svm_problem &prob, struct svm_parameter &param, struct svm_node *&x_space)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename, "r");
	char *endptr;
	char *idx, *val, *label;

	if (fp == NULL)
	{
		std::cout << "Can't open input file " << filename << std::endl;
		return false;
	}

	prob.l = 0;
	elements = 0;

	int max_line_len = 1024;
	char *line = (char *)malloc(sizeof(char) * max_line_len);
	while (readline(fp, line, max_line_len) != NULL)
	{
		char *p = strtok(line, " \t");  // Label.

		// features
		while (1)
		{
			p = strtok(NULL, " \t");
			if (p == NULL || *p == '\n')  // Check '\n' as ' ' may be after the last feature.
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
		inst_max_index = -1;  // Strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0.
		readline(fp, line, max_line_len);
		prob.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if (label == NULL)  // Empty line.
		{
			exit_input_error(i + 1);
			return false;
		}

		prob.y[i] = strtod(label, &endptr);
		if (endptr == label || *endptr != '\0')
		{
			exit_input_error(i + 1);
			return false;
		}

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int)strtol(idx, &endptr,10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
			{
				exit_input_error(i + 1);
				return false;
			}
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
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

	if (param.kernel_type == PRECOMPUTED)
		for (i = 0; i < prob.l; ++i)
		{
			if (prob.x[i][0].index != 0)
			{
				std::cout << "Wrong input format: first column must be 0:sample_serial_number" << std::endl;
				return false;
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				std::cout << "Wrong input format: sample_serial_number out of range" << std::endl;
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
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;  // Set degree in kernel function.
	param.gamma = 0;  // Set gamma in kernel function. default: 1 / num_features.
	param.coef0 = 0;  // Set coef0 in kernel function.
	param.nu = 0.5;  // Set the parameter nu of nu-SVC, one-class SVM, and nu-SVR.
	param.cache_size = 100;  // Set cache memory size in MB.
	param.C = 1.0;  // Set the parameter C of C-SVC, epsilon-SVR, and nu-SVR.
	param.eps = 1e-3;  // Set tolerance of termination criterion.
	param.p = 0.1;  // Set the epsilon in loss function of epsilon-SVR.
	param.shrinking = 1;  // Whether to use the shrinking heuristics, 0 or 1.
	param.probability = 0;  // Whether to train a SVC or SVR model for probability estimates, 0 or 1.
	param.nr_weight = 0;  // Set the parameter C of class i to weight*C, for C-SVC.
	param.weight_label = NULL;  // Set the parameter C of class i to weight*C, for C-SVC.
	param.weight = NULL;  // Set the parameter C of class i to weight*C, for C-SVC.
	const bool cross_validation = false;
	const int nr_fold = 10;

	svm_set_print_string_function(local::print_null);

	const std::string input_file_name("./data/machine_learning/svm/heart_scale");
	const std::string model_file_name("./data/machine_learning/svm/heart_scale.model");

	struct svm_problem prob;
	struct svm_node *x_space = NULL;
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
