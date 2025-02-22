#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <crfpp/crfpp.h>
#else
#include <crfpp.h>
#endif
#include <iostream>
#include <string>
#include <cstdlib>


namespace {
namespace local {

bool example()
{
	// -v 3: access deep information like alpha, beta, prob.
	// -nN: enable nbest output. N should be >= 2.
	CRFPP::Tagger *tagger = CRFPP::createTagger("-m ./data/probabilistic_graphical_model/crfpp/model -v 3 -n2");

	if (!tagger)
	{
		std::cerr << "CRF++ error: " << CRFPP::getTaggerError() << std::endl;
		return false;
	}

	// Clear internal context.
	tagger->clear();

	// Add context.
	tagger->add("Confidence NN");
	tagger->add("in IN");
	tagger->add("the DT");
	tagger->add("pound NN");
	tagger->add("is VBZ");
	tagger->add("widely RB");
	tagger->add("expected VBN");
	tagger->add("to TO");
	tagger->add("take VB");
	tagger->add("another DT");
	tagger->add("sharp JJ");
	tagger->add("dive NN");
	tagger->add("if IN");
	tagger->add("trade NN");
	tagger->add("figures NNS");
	tagger->add("for IN");
	tagger->add("September NNP");

	std::cout << "Column size: " << tagger->xsize() << std::endl;
	std::cout << "Token size: " << tagger->size() << std::endl;
	std::cout << "Tag size: " << tagger->ysize() << std::endl;

	std::cout << "Tagset information:" << std::endl;
	for (size_t i = 0; i < tagger->ysize(); ++i)
	{
		std::cout << "\tTag " << i << " " << tagger->yname(i) << std::endl;
	}

	// Parse and change internal stated as 'parsed'.
	if (!tagger->parse()) return false;

	std::cout << "Conditional prob = " << tagger->prob() << ", log(Z) = " << tagger->Z() << std::endl;

	for (size_t i = 0; i < tagger->size(); ++i)
	{
		for (size_t j = 0; j < tagger->xsize(); ++j)
		{
			std::cout << tagger->x(i, j) << '\t';
		}
		std::cout << tagger->y2(i) << '\t' << std::endl;

		std::cout << "Details:";
		for (size_t j = 0; j < tagger->ysize(); ++j)
		{
			std::cout << '\t' << tagger->yname(j) << "/prob=" << tagger->prob(i, j) << "/alpha=" << tagger->alpha(i, j) << "/beta=" << tagger->beta(i, j);
		}
		std::cout << std::endl;
	}

	// When -n20 is specified, you can access nbest outputs.
	std::cout << "nbest outputs: " << std::endl;
	for (size_t n = 0; n < 10; ++n)
	{
		if (!tagger->next()) break;
		std::cout << "nbest n =" << n << ",\tconditional prob = " << tagger->prob() << std::endl;
		// You can access any information using tagger->y()...
	}
	std::cout << "done" << std::endl;

	delete tagger;

	return true;
}

}  // namespace local
}  // unnamed namespace

namespace my_crfpp {

}  // namespace my_crfpp

int crfpp_main(int argc, char *argv[])
{
	//const std::string base_directory("./data/probabilistic_graphical_model/crfpp/basenp/");
	//const std::string base_directory("./data/probabilistic_graphical_model/crfpp/chunking/");
	//const std::string base_directory("./data/probabilistic_graphical_model/crfpp/JapaneseNE/");
	const std::string base_directory("./data/probabilistic_graphical_model/crfpp/seg/");

	int retval = EXIT_SUCCESS;

	// Train (encoding).
	std::cout << "Training (encoding) ..." << std::endl;
	{
		const std::string template_filename(base_directory + "template");
		const std::string training_data_filename(base_directory + "train.data");
		const std::string model_filename(base_directory + "model");

#if 1
		// crf_learn -a CRF-L1 -f 3 -c 4.0 -p 4 template train.data model
		const int my_argc = 12;
		const char *my_argv[my_argc] = {
			argv[0],
			"-a", "CRF-L1", "-f", "3", "-c", "4.0", "-p", "4",
			template_filename.c_str(), training_data_filename.c_str(), model_filename.c_str()
		};
#elif 0
		// crf_learn -a CRF-L2 -f 3 -c 4.0 -p 4 template train.data model
		const int my_argc = 12;
		const char *my_argv[my_argc] = {
			argv[0],
			"-a", "CRF-L2", "-f", "3", "-c", "4.0", "-p", "4",
			template_filename.c_str(), training_data_filename.c_str(), model_filename.c_str()
		};
#elif 0
		// crf_learn -a MIRA -f 3 -p 4 template train.data model
		const int my_argc = 10;
		const char *my_argv[my_argc] = {
			argv[0],
			"-a", "MIRA", "-f", "3", "-p", "4",
			template_filename.c_str(), training_data_filename.c_str(), model_filename.c_str()
		};
#endif

		retval = crfpp_learn(my_argc, (char **)my_argv);
	}

	// Test (decoding).
	std::cout << "Testing (decoding) ..." << std::endl;
	{
		const std::string model_filename(base_directory + "model");
		const std::string testing_data_filename(base_directory + "test.data");

		// crf_test -m model test.data
		const int my_argc = 4;
		const char *my_argv[my_argc] = {
			argv[0],
			"-m", model_filename.c_str(), testing_data_filename.c_str()
		};

		retval = crfpp_test(my_argc, (char **)my_argv);
	}

	// Run example.
	std::cout << "Running example ..." << std::endl;
	{
		retval = local::example() ? EXIT_SUCCESS : EXIT_FAILURE;
	}

	return retval;
}
