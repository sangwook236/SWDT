//include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
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
	int tiny_dnn_main(int argc, char *argv[]);
	int manifold_learning_main(int argc, char *argv[]);
	int manifold_alignment_main(int argc, char *argv[]);
	int libgp_main(int argc, char *argv[]);

	int rl_glue_main(int argc, char *argv[]);
	int rllib_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)time(NULL));

		std::cout << "clustering algorithm ------------------------------------------------" << std::endl;
		//	- k-means & k-means++ algorithms.
		//	- Spectral clustering.
		//retval = clustering_main(argc, argv);

		std::cout << "\nlibsvm library ------------------------------------------------------" << std::endl;
		//retval = libsvm_main(argc, argv);

		std::cout << "\nmysvm library -------------------------------------------------------" << std::endl;
		//retval = mysvm_main(argc, argv);  // Not yet implemented.

		std::cout << "\nSVM-Light library ---------------------------------------------------" << std::endl;
		//	- SVM struct: structured SVM.
		//	- SVM multiclass: multi-class SVM.
		//	- SVM hmm: structured SVMs for sequence tagging.
		//	- SVM alignment: structured SVMs for sequence alignment.
		//	- Latent SVM struct: latent structured SVM.
		//retval = svm_light_main(argc, argv);  // Not yet implemented.

		std::cout << "\nmultiboost library --------------------------------------------------" << std::endl;
		//retval = multiboost_main(argc, argv);  // Not yet implemented.

		std::cout << "\nshogun library ------------------------------------------------------" << std::endl;
		//	- Multiple kernel learning (MKL).
		//	- Gaussian process (GP) regression.
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = shogun_main(argc, argv);
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nEncog Machine Learning Framework ------------------------------------" << std::endl;
		//	- Java, .NET and C/C++.
		//	- Meural network (NN).
		//		ADALINE neural network.
		//		Adaptive resonance theory 1 (ART1).
		//		Bidirectional associative memory (BAM).
		//		Boltzmann machine.
		//		Feedforward neural network.
		//		Recurrent neural network.
		//		Hopfield neural network.
		//		Radial basis function network (RBFN).
		//		Nuroevolution of augmenting topologies (NEAT).
		//		(Recurrent) self organizing map (SOM).
		//retval = encog_main(argc, argv);  // Not yet implemented.

		std::cout << "\ntorch library -------------------------------------------------------" << std::endl;
		//  - Tensor.
		//	- Deep learning.
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = torch_main(argc, argv);  // Not yet implemented.
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\ndlib-ml library -----------------------------------------------------" << std::endl;
		//	- Support vector machines (SVM).
		//	- Relevance vector machines (RVM).
		//	- Structured prediction.
		//	- Multi-layer perceptrons (MLP).
		//	- Radial basis function network (RBFN).
		//	- Clustering.
		//	- Unsupervised learning.
		//		Canonical correlation analysis (CCA).
		//	- Semi-supervised learning.
		//	- Feature selection.
		//	- Optimization.
		//	- Graphical model.
		//		Bayesian network.
		//		Inference algorithms.
		//	- Image processing.
		//	- Reinforcement learning (RL).
		//		Least-squares policy iteration (LSPI).
		// REF [library] >> ${SWDT_C++_HOME}/ext/test/general_purpose_library/dlib.

		std::cout << "\nliblearning library -------------------------------------------------" << std::endl;
		//	- Deep learning.
		//retval = liblearning_main(argc, argv);

		std::cout << "\nWaffles library -----------------------------------------------------" << std::endl;
		//	- Generation of various types of data.
		//	- Supervised and semi-supervised learning algorithms.
		//	- Transforming datasets.
		//	- Predicting missing values in incomplete data, or testing collaborative filtering recommendation systems.
		//	- Learning from and operating on sparse data.
		//	- Plotting and visualizing datasets.
		//	- Audio processing.
		//	- Clustering.
		//	- Dimensionality reduction, manifold learning, attribute selection, and tools related to NLDR.
		//retval = waffles_main(argc, argv);

		std::cout << "\nCaffe framework -----------------------------------------------------" << std::endl;
		//	- Deep learning.
		//retval = caffe_main(argc, argv);

		std::cout << "\ntiny-dnn library ----------------------------------------------------" << std::endl;
		//	- Deep learning.
		//		Convolutional neural network.
		//		Denoising auto-encoder.
		//		Dropout.
		retval = tiny_dnn_main(argc, argv);

		std::cout << "\nManifold learning & alignment ---------------------------------------" << std::endl;
		//retval = manifold_learning_main(argc, argv);  // Not yet implemented.
		//retval = manifold_alignment_main(argc, argv);

		std::cout << "\nlibgp library ------------------------------------------------------" << std::endl;
		//retval = libgp_main(argc, argv);

		std::cout << "\nRL-Glue (Reinforcement Learning Glue) library -----------------------" << std::endl;
		//retval = rl_glue_main(argc, argv);  // Not yet implemented.

		std::cout << "\nRLlib library -------------------------------------------------------" << std::endl;
		//retval = rllib_main(argc, argv);
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
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
