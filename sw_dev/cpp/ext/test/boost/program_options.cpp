#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <vector>


namespace {
//namespace local {

template<class T>
std::ostream & operator<<(std::ostream &os, const std::vector<T> &v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
	return os;
}

//}  // namespace local
}  // unnamed namespace

bool program_options(int argc, char *argv[])
{
	// Usage:
	//	foo --setting.serial.baud-rate=19200 --include-path=/usr/local/include file1.txt
	//	foo -B 19200 -I /usr/local/include file1.txt

	//---------------------------------------------------------------------------------------------
	// Step #1: Declare a group of options that will be allowed only on command line.

	boost::program_options::options_description general("Generic options");
	general.add_options()
		("version,v", "Print version string")
		("help,h", "Produce help message")
		;

	// Declare a group of options that will be allowed both on command line and in config file.
	int opt;
	boost::program_options::options_description config("Configuration");
	config.add_options()
		("optimization", boost::program_options::value<int>(&opt)->default_value(10), "Optimization level")
		// composing(): compose both inputs in command line and in config file.
		// Otherwise, inputs of command line have priority.
		("include-path,I", boost::program_options::value<std::vector<std::string> >()->composing(), "Include path")
		;

	//
	bool opt_available;
	//unsigned int opt_baudrate;
	boost::program_options::options_description setting("Setting");
	setting.add_options()
		("setting.serial.available,A", boost::program_options::value<bool>(&opt_available)->default_value(false), "Is available")
		("setting.serial.port,P", boost::program_options::value<std::string>()->default_value("COM4"), "COM port name")
		("setting.serial.baud-rate,B", boost::program_options::value<unsigned int>()->default_value(9600), "Baud rate")
		;

	// Hidden options, will be allowed both on command line and in config file, but will not be shown to the user.
	boost::program_options::options_description hidden("Hidden options");
	hidden.add_options()
		("input-file", boost::program_options::value<std::vector<std::string> >(), "Input file")
		;

	//---------------------------------------------------------------------------------------------
	// Step #2

	boost::program_options::options_description cmdline_options;
	cmdline_options.add(general).add(config).add(setting).add(hidden);

	boost::program_options::options_description config_file_options;
	config_file_options.add(config).add(setting).add(hidden);

	//boost::program_options::options_description visible_options("Allowed options");
	boost::program_options::options_description visible_options;
	visible_options.add(general).add(config).add(setting);

	//---------------------------------------------------------------------------------------------
	// Step #3: Positional option.
	//	foo --optimization=9 /etc/passwd
	//		Here, the "/etc/passwd" element does not have any option name ==> positional option.

	boost::program_options::positional_options_description positional_options;
	positional_options.add("input-file", -1);

	//---------------------------------------------------------------------------------------------
	// Step #4

	boost::program_options::variables_map vm;

	//---------------------------------------------------------------------------------------------
	// From command line.
	// --optimization=9 --input-file=/etc/passwd1 --input-file=/etc/passwd2 /etc/var/passwd3

#if defined(_UNICODE) || defined(UNICODE)
	//boost::program_options::store(boost::program_options::wcommand_line_parser(argc, argv).options(cmdline_options).positional(positional_options).run(), vm);
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(cmdline_options).positional(positional_options).run(), vm);
#else
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(cmdline_options).positional(positional_options).run(), vm);
#endif
	//boost::program_options::store(boost::program_options::parse_command_line(argc, argv, cmdline_options), vm);

	//---------------------------------------------------------------------------------------------
	// From file.

	std::ifstream stream("data/boost/boost_multiple_sources.cfg");
	if (stream.is_open())
	{
		boost::program_options::store(boost::program_options::parse_config_file(stream, config_file_options), vm);
		stream.close();
	}

	boost::program_options::notify(vm);

	//---------------------------------------------------------------------------------------------
	// Step #5

	if (vm.count("help"))
	{
		std::cout << visible_options << std::endl;
		//return true;
		return false;
	}

	if (vm.count("version"))
	{
		std::cout << "Multiple sources example, version 1.00.00" << std::endl;
		//return true;
		return false;
	}

	//---------------------------------------------------------------------------------------------
	// Display.
	{
		//const int optimization = vm["optimization"].as<int>();
		//const std::vector<std::string> include_paths = vm["include-path"].as<std::vector<std::string> >();

		//const std::vector<std::string> input_files = vm["input-file"].as<std::vector<std::string> >();

		//const bool serial_available = vm["setting.serial.available"].as<int>();
		//const std::string serial_port = vm["setting.serial.port"].as<std::string>();
		//const unsigned int serial_baud_rate = vm["setting.serial.baud-rate"].as<unsigned int>();
	}

	{
		const std::size_t num = vm.count("include-path");
		std::cout << "The number of include paths is " << num << std::endl;
		if (vm.count("include-path"))
			std::cout << "Include paths are: " << vm["include-path"].as<std::vector<std::string> >() << std::endl;

		if (vm.count("input-file"))
			std::cout << "Input files are: " << vm["input-file"].as<std::vector<std::string> >() << std::endl;

		std::cout << "Optimization level is " << opt << std::endl;

		std::cout << "Serial Comm is " << (opt_available ? "available" : "not available") << std::endl;
		if (vm.count("setting.serial.port"))
			std::cout << "Serial Comm Port is " << vm["setting.serial.port"].as<std::string>() << std::endl;
		//std::cout << "Serial Comm Baud rate is " << opt_baudrate << std::endl;
		if (vm.count("setting.serial.baud-rate"))
			std::cout << "Serial Comm Baud rate is " << vm["setting.serial.baud-rate"].as<unsigned int>() << std::endl;
	}

	return true;
}
