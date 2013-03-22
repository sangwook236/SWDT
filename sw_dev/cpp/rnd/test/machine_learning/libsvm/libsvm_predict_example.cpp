#include "../libsvm_lib/svm.h"
#include <iostream>
#include <string>


namespace {
namespace local {

int print_null(const char *s)
{
	return 0;
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

bool predict(FILE *input, FILE *output, const struct svm_model *model, const bool predict_probability, int &max_nr_attr, struct svm_node *x)
{
	const int svm_type = svm_get_svm_type(model);
	const int nr_class = svm_get_nr_class(model);

	double *prob_estimates = NULL;
	if (predict_probability)
	{
		if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
			std::cout << "Prob. model for test data: target value = predicted value + z," << std::endl
				<< "z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma = " << svm_get_svr_probability(model) << std::endl;
		else
		{
			int *labels = (int *)malloc(nr_class * sizeof(int));
			svm_get_labels(model, labels);
			prob_estimates = (double *)malloc(nr_class * sizeof(double));
			fprintf(output, "labels");		
			for (int j = 0; j < nr_class; ++j)
				fprintf(output, " %d", labels[j]);
			fprintf(output, "\n");
			free(labels);
		}
	}

	int max_line_len = 1024;
	char *line = (char *)malloc(max_line_len * sizeof(char));
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	while (readline(input, line, max_line_len) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1;  // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line, " \t\n");
		if (label == NULL)  // empty line
		{
			exit_input_error(total + 1);
			return false;
		}

		target_label = strtod(label, &endptr);
		if (endptr == label || *endptr != '\0')
		{
			exit_input_error(total + 1);
			return false;
		}

		while (1)
		{
			if (i >= max_nr_attr - 1)  // need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *)realloc(x, max_nr_attr * sizeof(struct svm_node));
			}

			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;
			errno = 0;
			x[i].index = (int)strtol(idx, &endptr,10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
			{
				exit_input_error(total + 1);
				return false;
			}
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
			{
				exit_input_error(total+1);
				return false;
			}

			++i;
		}
		x[i].index = -1;

		if (predict_probability && (svm_type == C_SVC || svm_type == NU_SVC))
		{
			predict_label = svm_predict_probability(model, x, prob_estimates);
			fprintf(output, "%g", predict_label);
			for (int j = 0; j < nr_class; ++j)
				fprintf(output, " %g", prob_estimates[j]);
			fprintf(output, "\n");
		}
		else
		{
			predict_label = svm_predict(model, x);
			fprintf(output, "%g\n", predict_label);
		}

		if (predict_label == target_label)
			++correct;
		error += (predict_label - target_label) * (predict_label - target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label * predict_label;
		sumtt += target_label * target_label;
		sumpt += predict_label * target_label;
		++total;
	}
	if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
	{
		std::cout << "Mean squared error = " << (error / total) << " (regression)" << std::endl;
		std::cout << "Squared correlation coefficient = " << ((total*sumpt - sump*sumt) * (total*sumpt - sump*sumt)) / ((total*sumpp - sump*sump) * (total*sumtt - sumt*sumt)) << " (regression)" << std::endl;
	}
	else
		std::cout << "Accuracy = " << ((double)correct / total * 100) << "% (" << correct << '/' << total << ") (classification)" << std::endl;

	if (predict_probability)
		free(prob_estimates);

	free(line);
	return true;
}

}  // namespace local
}  // unnamed namespace

namespace my_libsvm {

// [ref] ${LIBSVM_HOME}/svm-predict.c
void predict_example()
{
	const bool predict_probability = false;

	const std::string test_file_name("./machine_learning_data/svm/heart_scale");
	const std::string output_file_name("./machine_learning_data/svm/heart_scale.output");
	const std::string model_file_name("./machine_learning_data/svm/heart_scale.model");

	FILE *input = fopen(test_file_name.c_str(), "r");
	if(input == NULL)
	{
		std::cout << "can't open input file " << test_file_name << std::endl;
		return;
	}

	FILE *output = fopen(output_file_name.c_str(), "w");
	if(output == NULL)
	{
		std::cout << "can't open output file " << output_file_name << std::endl;
		return;
	}

	struct svm_model *model = svm_load_model(model_file_name.c_str());
	if (NULL == model)
	{
		std::cout << "can't open model file " << model_file_name << std::endl;
		return;
	}

	if (predict_probability)
	{
		if (svm_check_probability_model(model) == 0)
		{
			std::cout << "Model does not support probabiliy estimates" << std::endl;
			return;
		}
	}
	else
	{
		if (svm_check_probability_model(model) != 0)
			std::cout << "Model supports probability estimates, but disabled in prediction." << std::endl;
	}

	int max_nr_attr = 64;
	struct svm_node *x = (struct svm_node *)malloc(max_nr_attr * sizeof(struct svm_node));
	local::predict(input, output, model, predict_probability, max_nr_attr, x);

	free(x);
	fclose(input);
	fclose(output);

	svm_free_and_destroy_model(&model);
}

}  // namespace my_libsvm
