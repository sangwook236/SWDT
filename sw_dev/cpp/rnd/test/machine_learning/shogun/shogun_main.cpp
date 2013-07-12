//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/lib/ShogunException.h>
#include <iostream>


namespace {
namespace local {

void print_message(FILE *target, const char *str)
{
	fprintf(target, "%s", str);
}

void print_warning(FILE *target, const char *str)
{
	fprintf(target, "%s", str);
}

void print_error(FILE *target, const char *str)
{
	fprintf(target, "%s", str);
}

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

void clustering_kmeans_example();

void classifier_conjugateindex_example();
void classifier_minimal_svm_example();
void classifier_latent_svm_example();
void classifier_libsvm_example();
void classifier_multiclass_ecoc_example();
void classifier_multiclass_ecoc_discriminant_example();
void classifier_multiclass_ecoc_random_example();
void classifier_multiclass_shareboost_example();
void classifier_multiclasslibsvm_example();
void classifier_multiclasslinearmachine_example();
void classifier_mklmulticlass_example();

void regression_gaussian_process_ard_example();
void regression_gaussian_process_fitc_example();
void regression_gaussian_process_gaussian_example();
void regression_gaussian_process_laplace_example();
void regression_gaussian_process_product_example();
void regression_gaussian_process_simple_exact_example();
void regression_gaussian_process_sum_example();

void converter_factoranalysis_example();
void converter_multidimensionalscaling_example();

void converter_isomap_example();
void converter_locallylinearembedding_example();
void converter_kernellocallylinearembedding_example();
void converter_diffusionmaps_example();
void converter_laplacianeigenmaps_example();
void converter_hessianlocallylinearembedding_example();
void converter_localtangentspacealignment_example();
void converter_linearlocaltangentspacealignment_example();
void converter_localitypreservingprojections_example();
void converter_neighborhoodpreservingembedding_example();
void converter_stochasticproximityembedding_example();

void evaluation_cross_validation_classification_example();
void evaluation_cross_validation_regression_example();
void evaluation_cross_validation_multiclass_example();
void evaluation_cross_validation_multiclass_mkl_example();
void evaluation_cross_validation_mkl_weight_storage_example();

void modelselection_grid_search_linear_example();
void modelselection_grid_search_kernel_example();
void modelselection_grid_search_multiclass_svm_example();
void modelselection_grid_search_mkl_example();
void modelselection_apply_parameter_tree_example();
void modelselection_grid_search_krr_example();

}  // namespace my_shogun

int shogun_main(int argc, char *argv[])
{
#if 1
	shogun::init_shogun_with_defaults();
#else
	shogun::init_shogun(&local::print_message, &local::print_warning, &local::print_error);
#endif

	int retval = 0;
	try
	{
		//shogun::sg_version->print_version();
		//shogun::sg_io->set_loglevel(MSG_INFO);

		std::cout << "shogun library: clustering example ------------------------------" << std::endl;
		if (false)
		{
            my_shogun::clustering_kmeans_example();
		}

		std::cout << "\nshogun library: classification example (kernel machine) ---------" << std::endl;
 		if (false)
		{
            //my_shogun::classifier_conjugateindex_example();

            //my_shogun::classifier_minimal_svm_example();
            //my_shogun::classifier_latent_svm_example();  // not yet implemented
            //my_shogun::classifier_libsvm_example();

            //my_shogun::classifier_multiclasslibsvm_example();
            //my_shogun::classifier_multiclasslinearmachine_example();

            //my_shogun::classifier_multiclass_ecoc_example();
            //my_shogun::classifier_multiclass_ecoc_discriminant_example();
            //my_shogun::classifier_multiclass_ecoc_random_example();

            my_shogun::classifier_mklmulticlass_example();
		}

		std::cout << "\nshogun library: classification example (boosting) ---------------" << std::endl;
 		if (false)
		{
            my_shogun::classifier_multiclass_shareboost_example();
		}

		std::cout << "\nshogun library: Gaussian process (GP) regression example --------" << std::endl;
  		if (false)
		{
            my_shogun::regression_gaussian_process_simple_exact_example();
            my_shogun::regression_gaussian_process_sum_example();
            my_shogun::regression_gaussian_process_product_example();

			my_shogun::regression_gaussian_process_ard_example();
            my_shogun::regression_gaussian_process_fitc_example();
            my_shogun::regression_gaussian_process_gaussian_example();
            my_shogun::regression_gaussian_process_laplace_example();
		}

		std::cout << "\nshogun library: linear dimensionality reduction example ---------" << std::endl;
 		if (false)
		{
            my_shogun::converter_factoranalysis_example();
            my_shogun::converter_multidimensionalscaling_example();
		}

		std::cout << "\nshogun library: nonlinear dimensionality reduction example ------" << std::endl;
  		if (false)
		{
            my_shogun::converter_isomap_example();
            my_shogun::converter_locallylinearembedding_example();
            my_shogun::converter_kernellocallylinearembedding_example();
            my_shogun::converter_diffusionmaps_example();
            my_shogun::converter_laplacianeigenmaps_example();
            my_shogun::converter_hessianlocallylinearembedding_example();
            my_shogun::converter_localtangentspacealignment_example();
            my_shogun::converter_linearlocaltangentspacealignment_example();

            my_shogun::converter_localitypreservingprojections_example();
            my_shogun::converter_neighborhoodpreservingembedding_example();
            my_shogun::converter_stochasticproximityembedding_example();
		}

 		std::cout << "\nshogun library: cross validation example ------------------------" << std::endl;
   		if (true)
		{
            my_shogun::evaluation_cross_validation_classification_example();
            //my_shogun::evaluation_cross_validation_regression_example();

            //my_shogun::evaluation_cross_validation_multiclass_example();
            //my_shogun::evaluation_cross_validation_multiclass_mkl_example();
            //my_shogun::evaluation_cross_validation_mkl_weight_storage_example();
		}

 		std::cout << "\nshogun library: model selection example -------------------------" << std::endl;
   		if (false)
		{
			 my_shogun::modelselection_grid_search_linear_example();
			 //my_shogun::modelselection_grid_search_kernel_example();

			 //my_shogun::modelselection_grid_search_multiclass_svm_example();
			 //my_shogun::modelselection_grid_search_mkl_example();

			 //my_shogun::modelselection_apply_parameter_tree_example();
			 //my_shogun::modelselection_grid_search_krr_example();
		}
	}
	catch (const shogun::ShogunException &e)
	{
		std::cout << "shogun::ShogunException caught: " << const_cast<shogun::ShogunException &>(e).get_exception_string() << std::endl;
		retval = 1;
	}

	shogun::exit_shogun();

	return retval;
}
