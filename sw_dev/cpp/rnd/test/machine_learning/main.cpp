//include "stdafx.h"
#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int clustering_main(int argc, char *argv[]);

	int libsvm_main(int argc, char *argv[]);
	int mysvm_main(int argc, char *argv[]);
	int svm_light_main(int argc, char *argv[]);

	int multiboost_main(int argc, char *argv[]);

	int shogun_main(int argc, char *argv[]);
	int encog_main(int argc, char *argv[]);
	int torch_main(int argc, char *argv[]);
	int liblearning_main(int argc, char *argv[]);
	int waffles_main(int argc, char *argv[]);

	int caffe_main(int argc, char *argv[]);
	int tiny_cnn_main(int argc, char *argv[]);

	int manifold_learning_main(int argc, char *argv[]);
	int manifold_alignment_main(int argc, char *argv[]);

	int libgp_main(int argc, char *argv[]);

	int rl_glue_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)time(NULL));

		std::cout << "clustering algorithm ------------------------------------------------" << std::endl;
		//	-. k-means & k-means++ algorithms.
		//	-. spectral clustering.
		//retval = clustering_main(argc, argv);

		std::cout << "\nlibsvm library ------------------------------------------------------" << std::endl;
		//retval = libsvm_main(argc, argv);

		std::cout << "\nmysvm library -------------------------------------------------------" << std::endl;
		//retval = mysvm_main(argc, argv);  // not yet implemented.

		std::cout << "\nSVM-Light library ---------------------------------------------------" << std::endl;
		//	-. SVM struct: structured SVM.
		//	-. SVM multiclass: multi-class SVM.
		//	-. SVM hmm: structured SVMs for sequence tagging.
		//	-. SVM alignment: structured SVMs for sequence alignment.
		//	-. latent SVM struct: latent structured SVM.
		//retval = svm_light_main(argc, argv);  // not yet implemented.

		std::cout << "\nmultiboost library --------------------------------------------------" << std::endl;
		//retval = multiboost_main(argc, argv);  // not yet implemented.

		std::cout << "\nshogun library ------------------------------------------------------" << std::endl;
		//	-. multiple kernel learning (MKL).
		//	-. Gaussian process (GP) regression.
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = shogun_main(argc, argv);
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nEncog Machine Learning Framework ------------------------------------" << std::endl;
		//	-. Java, .NET and C/C++.
		//	-. neural network.
		//		ADALINE neural network.
		//		adaptive resonance theory 1 (ART1).
		//		bidirectional associative memory (BAM).
		//		Boltzmann machine.
		//		feedforward neural network.
		//		recurrent neural network.
		//		Hopfield neural network.
		//		radial basis function network (RBFN).
		//		neuroevolution of augmenting topologies (NEAT).
		//		(recurrent) self organizing map (SOM).
		//retval = encog_main(argc, argv);  // not yet implemented.

		std::cout << "\ntorch library -------------------------------------------------------" << std::endl;
		//  -. tensor.
		//	-. deep learning.
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = torch_main(argc, argv);  // not yet implemented.
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\ndlib-ml library -----------------------------------------------------" << std::endl;
		//	-. support vector machines (SVM).
		//	-. relevance vector machines (RVM).
		//	-. structured prediction.
		//	-. multi-layer perceptrons (MLP).
		//	-. radial basis function network (RBFN).
		//	-. clustering.
		//	-. unsupervised learning.
		//		canonical correlation analysis (CCA).
		//	-. semi-supervised learning.
		//	-. feature selection.
		//	-. optimization.
		//	-. graphical model.
		//		Bayesian network.
		//		inference algorithms.
		//	-. image processing.
		//	-. reinforcement learning (RL).
		//		least-squares policy iteration (LSPI).
		// REF [library] >> ${GDT_HOME}/sw_dev/cpp/ext/src/general_purpose_library/dlib.

		std::cout << "\nliblearning library -------------------------------------------------" << std::endl;
		//	-. deep learning.
		//retval = liblearning_main(argc, argv);

		std::cout << "\nWaffles library -----------------------------------------------------" << std::endl;
		//	-. generation of various types of data.
		//	-. supervised and semi-supervised learning algorithms.
		//	-. transforming datasets.
		//	-. predicting missing values in incomplete data, or testing collaborative filtering recommendation systems.
		//	-. learning from and operating on sparse data.
		//	-. plotting and visualizing datasets.
		//	-. audio processing.
		//	-. clustering.
		//	-. dimensionality reduction, manifold learning, attribute selection, and tools related to NLDR.
		//retval = waffles_main(argc, argv);

		std::cout << "\nCaffe framework -----------------------------------------------------" << std::endl;
		//	-. deep learning.
		//retval = caffe_main(argc, argv);

		std::cout << "\ntiny-cnn library ----------------------------------------------------" << std::endl;
		//	-. deep learning.
		retval = tiny_cnn_main(argc, argv);

		std::cout << "\nManifold learning & alignment ---------------------------------------" << std::endl;
		//retval = manifold_learning_main(argc, argv);  // not yet implemented.
		//retval = manifold_alignment_main(argc, argv);

		std::cout << "\nlibgp library ------------------------------------------------------" << std::endl;
		retval = libgp_main(argc, argv);

		std::cout << "\nRL-Glue (Reinforcement Learning Glue) library -----------------------" << std::endl;
		//retval = rl_glue_main(argc, argv);  // not yet implemented.
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
