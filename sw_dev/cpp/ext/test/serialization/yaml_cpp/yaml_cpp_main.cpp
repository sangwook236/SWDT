//#include "stdafx.h"
#include <yaml-cpp/yaml.h>
#include <iostream>


namespace {
namespace local {

void simple_example()
{
	const std::string config_filepath("./data/serialization/yaml/infer_svdd.yaml");
	//const std::string config_filepath("./data/serialization/yaml/infer_oe.yaml");
	//const std::string config_filepath("./data/serialization/yaml/infer_byol.yaml");

	const YAML::Node config(YAML::LoadFile(config_filepath));

	//-----
	if (config["model_name"])
		std::cout << "Model name = " << config["model_name"].as<std::string>() << std::endl;
	if (config["ssl_type"])
		std::cout << "SSL type = " << config["ssl_type"].as<std::string>() << std::endl;

	std::cout << "Stage = " << config["stage"].as<std::string>() << std::endl;

	//-----
	const YAML::Node data_dirs(config["data"]["data_dirs"]);
	std::cout << "Data dir:" << std::endl;
	for (const auto &elem: data_dirs)
		std::cout << '\t' << elem.as<std::string>() << std::endl;
	const YAML::Node image_shape(config["data"]["image_shape"]);
	std::cout << "Image shape: ";
	for (const auto &elem: image_shape)
		std::cout << elem.as<int>() << ", ";
	std::cout << std::endl;
	const YAML::Node image_roi(config["data"]["image_roi"]);
	std::cout << "Image RoI: ";
	for (const auto &elem: image_roi)
		std::cout << elem.as<int>() << ", ";
	std::cout << std::endl;
	std::cout << "Is image preloaded = " << config["data"]["is_image_preloaded"].as<bool>() << std::endl;

	std::cout << "Batch size = " << config["data"]["batch_size"].as<int>() << std::endl;
	std::cout << "#workers = " << config["data"]["num_workers"].as<int>() << std::endl;

	//-----
	const YAML::Node model(config["model"]);
	if (model["anomaly_score_threshold"])
		std::cout << "Anomaly score threshold = " << model["anomaly_score_threshold"].as<float>() << std::endl;
	else
		std::cout << "No anomaly score threshold." << std::endl;

	if (model["use_projector"])
		std::cout << "Use projector = " << (model["use_projector"].as<bool>() ? "true" : "false") << std::endl;
	else
		std::cout << "No use projector." << std::endl;
	if (model["use_predictor"])
		std::cout << "Use predictor = " << (model["use_predictor"].as<bool>() ? "true" : "false") << std::endl;
	else
		std::cout << "No use predictor." << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_yaml_cpp {

#if 0
// For the old APIs.
void basic_parsing();
void basic_emitting();
#else
void configuration_example();
void example_0_5();
#endif

}  // namespace my_yaml_cpp

int yaml_cpp_main(int argc, char *argv[])
{
	try
	{
#if 0
		// For the old APIs.
		//my_yaml_cpp::basic_parsing();
		//my_yaml_cpp::basic_emitting();
#else
		//my_yaml_cpp::configuration_example();
		//my_yaml_cpp::example_0_5();
#endif

		local::simple_example();
	}
	catch (const YAML::RepresentationException &ex)
	{
		std::cerr << "YAML::RepresentationException caught: " << ex.what() << std::endl;
		return 1;
	}
	catch (const YAML::ParserException &ex)
	{
		std::cerr << "YAML::ParserException caught: " << ex.what() << std::endl;
		return 1;
	}
	catch (const YAML::EmitterException &ex)
	{
		std::cerr << "YAML::EmitterException caught: " << ex.what() << std::endl;
		return 1;
	}
	catch (const YAML::Exception &ex)
	{
		std::cerr << "YAML::Exception caught: " << ex.what() << std::endl;
		return 1;
	}

	return 0;
}
