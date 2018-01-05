#include <dlib/svm.h>
#include <iostream>
#include <vector>
#include <cmath>


namespace {
namespace local {

typedef dlib::matrix<double, 2, 1> sample_type;
typedef dlib::radial_basis_kernel<sample_type> kernel_type;

class cross_validation_objective
{
public:
	cross_validation_objective(const std::vector<sample_type>& samples_, const std::vector<double>& labels_)
	: samples(samples_), labels(labels_)
	{}

	double operator() (const dlib::matrix<double>& params) const
	{
		// Pull out the two SVM model parameters.
		// Note that, in this case, I have setup the parameter search to operate in log scale so we have to remember to call exp() to put the parameters back into a normal scale.
		const double gamma = std::exp(params(0));
		const double nu = std::exp(params(1));

		// Make an SVM trainer and tell it what the parameters are supposed to be.
		dlib::svm_nu_trainer<kernel_type> trainer;
		trainer.set_kernel(kernel_type(gamma));
		trainer.set_nu(nu);

		// Perform 10-fold cross validation and then print and return the results.
		dlib::matrix<double> result = dlib::cross_validate_trainer(trainer, samples, labels, 10);
		std::cout << "gamma: " << std::setw(11) << gamma << "  nu: " << std::setw(11) << nu << "  cross validation accuracy: " << result;

		// Return the harmonic mean between the accuracies of each class.
		// However, you could do something else.
		// For example, you might care a lot more about correctly predicting the +1 class, so you could penalize results that didn't obtain a high accuracy on that class.
		// You might do this by using something like a weighted version of the F1-score (see http://en.wikipedia.org/wiki/F1_score).     
		return 2 * prod(result) / sum(result);
	}

	const std::vector<sample_type>& samples;
	const std::vector<double>& labels;
};

}  // namespace local
}  // unnamed namespace

namespace my_dlib {

// REF [file] >> ${DLIB_HOME}/examples/model_selection_ex.cpp
void model_selection_example()
{
	// Make objects to contain our samples and their respective labels.
	std::vector<local::sample_type> samples;
	std::vector<double> labels;

	// Put some data into our samples and labels objects.
	for (double r = -20; r <= 20; r += 0.8)
	{
		for (double c = -20; c <= 20; c += 0.8)
		{
			local::sample_type samp;
			samp(0) = r;
			samp(1) = c;
			samples.push_back(samp);

			// If this point is less than 10 from the origin.
			if (std::sqrt(r*r + c*c) <= 10)
				labels.push_back(+1);
			else
				labels.push_back(-1);
		}
	}

	std::cout << "Generated " << samples.size() << " points" << std::endl;

	// Normalize all the samples by subtracting their mean and dividing by their standard deviation.
	// This is generally a good idea since it often heads off numerical stability problems and also prevents one large feature from smothering others.
	// Doing this doesn't matter much in this example so I'm just doing this here so you can see an easy way to accomplish this with  the library.  
	dlib::vector_normalizer<local::sample_type> normalizer;
	// Let the normalizer learn the mean and standard deviation of the samples.
	normalizer.train(samples);
	// Normalize each sample.
	for (unsigned long i = 0; i < samples.size(); ++i)
		samples[i] = normalizer(samples[i]);

	// Randomize the order of the samples.
	dlib::randomize_samples(samples, labels);

	// The nu parameter has a maximum value that is dependent on the ratio of the +1 to -1 labels in the training data.
	// The 0.999 is here because the maximum allowable nu is strictly less than the value returned by maximum_nu().
	const double max_nu = 0.999 * dlib::maximum_nu(labels);

	// Simple grid search.
	double best_gamma = 0.1, best_nu;
	{
		// Generate a 4x4 grid of logarithmically spaced points.
		// The result is a matrix with 2 rows and 16 columns where each column represents one of our points. 
		dlib::matrix<double> params = dlib::cartesian_product(
			dlib::logspace(std::log10(5.0), std::log10(1e-5), 4),  // gamma parameter.
			dlib::logspace(std::log10(max_nu), std::log10(1e-5), 4)  // nu parameter.
		);
		// As an aside, if you wanted to do a grid search over points of dimensionality more than two you would just nest calls to cartesian_product().
		// You can also use linspace() to generate linearly spaced points if that is more appropriate for the parameters you are working with.   

		// Loop over all the points we generated and check how good each is.
		std::cout << "Doing a grid search" << std::endl;
		dlib::matrix<double> best_result(2, 1);
		best_result = 0;
		for (long col = 0; col < params.nc(); ++col)
		{
			// Pull out the current set of model parameters.
			const double gamma = params(0, col);
			const double nu = params(1, col);

			// Setup a training object using our current parameters.
			dlib::svm_nu_trainer<local::kernel_type> trainer;
			trainer.set_kernel(local::kernel_type(gamma));
			trainer.set_nu(nu);

			// Do 10 fold cross validation and then check if the results are the best we have seen so far.
			dlib::matrix<double> result = dlib::cross_validate_trainer(trainer, samples, labels, 10);
			std::cout << "gamma: " << std::setw(11) << gamma << "  nu: " << std::setw(11) << nu << "  cross validation accuracy: " << result;

			// Save the best results.
			if (dlib::sum(result) > dlib::sum(best_result))
			{
				best_result = result;
				best_gamma = gamma;
				best_nu = nu;
			}
		}

		std::cout << "\n Best result of grid search: " << dlib::sum(best_result) << std::endl;
		std::cout << " Best gamma: " << best_gamma << "   best nu: " << best_nu << std::endl;
	}

	// Use the BOBYQA algorithm.
	// It is a routine that performs optimization of a function in the absence of derivatives.  
	{
		std::cout << "\n\nTry the BOBYQA algorithm" << std::endl;

		// Supply a starting point for the optimization.
		// Here we are using the best result of the grid search.
		dlib::matrix<double> params;
		params.set_size(2, 1);
		params = best_gamma,  // Initial gamma.
			best_nu;  // Initial nu.

		// Supply lower and upper bounds for the search.  
		dlib::matrix<double> lower_bound(2, 1), upper_bound(2, 1);
		lower_bound = 1e-7,  // Smallest allowed gamma.
			1e-7;  // Smallest allowed nu.
		upper_bound = 100,  // Largest allowed gamma.
			max_nu;  // Largest allowed nu.

		// For the gamma and nu SVM parameters it is generally a good idea to search in log space.
		params = std::log(params);
		lower_bound = std::log(lower_bound);
		upper_bound = std::log(upper_bound);

		// Ask BOBYQA to look for the best set of parameters.
		// Note that we are using the cross validation function object defined at the top of the file.
		double best_score = dlib::find_max_bobyqa(
			local::cross_validation_objective(samples, labels),  // Function to maximize.
			params,  //	Starting point.
			params.size() * 2 + 1,  // See BOBYQA docs, generally size*2+1 is a good setting for this.
			lower_bound,  // Lower bound.
			upper_bound,  // Upper bound.
			dlib::min(upper_bound - lower_bound) / 10,  // Search radius.
			0.01,  // Desired accuracy.
			100  // Max number of allowable calls to cross_validation_objective().
		);

		// Don't forget to convert back from log scale to normal scale.
		params = std::exp(params);

		std::cout << " Best result of BOBYQA: " << best_score << std::endl;
		std::cout << " Best gamma: " << params(0) << "   best nu: " << params(1) << std::endl;

		// Note that the find_max_bobyqa() function only works for optimization problems with 2 variables or more.
		// If you only have a single variable then you should use the find_max_single_variable() function.
	}
}

}  // namespace my_dlib
