#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <vector>


template<class T>
std::ostream & operator<<(std::ostream& os, const std::vector<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " ")); 
	return os;
}

#if defined(_UNICODE) || defined(UNICODE)
bool program_options(int argc, wchar_t* argv[])
#else
bool program_options(int argc, char* argv[])
#endif
{
	//---------------------------------------------------------------------------------------------
	// step #1
	// Declare a group of options that will be allowed only on command line

	boost::program_options::options_description general("Generic options");
	general.add_options()
		("version,v", "print version string")
		("help", "produce help message")
		;

	// Declare a group of options that will be allowed both on command line and in config file
	int opt;
	boost::program_options::options_description config("Configuration");
	config.add_options()
		("optimization", boost::program_options::value<int>(&opt)->default_value(10), "optimization level")
		("include-path,I", boost::program_options::value<std::vector<std::string> >()->composing(), "include path")
		;

	//
	bool opt_available;
	//unsigned int opt_baudrate;
	boost::program_options::options_description setting("Setting");
	setting.add_options()
		("setting.serial.available,A", boost::program_options::value<bool>(&opt_available)->default_value(false), "is available")
		("setting.serial.port,P", boost::program_options::value<std::string>()->default_value("COM4"), "COM port name")
		("setting.serial.baud-rate,B", boost::program_options::value<unsigned int>()->default_value(9600), "Baud rate")
		;

	// Hidden options, will be allowed both on command line and in config file, but will not be shown to the user.
	boost::program_options::options_description hidden("Hidden options");
	hidden.add_options()
		("input-file", boost::program_options::value<std::vector<std::string> >(), "input file")
		;

	//---------------------------------------------------------------------------------------------
	// step #2

	boost::program_options::options_description cmdline_options;
	cmdline_options.add(general).add(config).add(setting).add(hidden);

	boost::program_options::options_description config_file_options;
	config_file_options.add(config).add(setting).add(hidden);

	boost::program_options::options_description visible("Allowed options");
	visible.add(general).add(config).add(setting);

	//---------------------------------------------------------------------------------------------
	// step #3: positional option
	// archiver --optimization=9 /etc/passwd
	// Here, the "/etc/passwd" element does not have any option name. ==> positional option

	boost::program_options::positional_options_description p;
	p.add("input-file", -1);

	//---------------------------------------------------------------------------------------------
	// step #4

	boost::program_options::variables_map vm;

	//---------------------------------------------------------------------------------------------
	// from command line
	// --optimization=9 --input-file=/etc/passwd1 --input-file=/etc/passwd2 /etc/var/passwd3

#if defined(_UNICODE) || defined(UNICODE)
	boost::program_options::store(boost::program_options::wcommand_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);
#else
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);
#endif
	//boost::program_options::store(boost::program_options::parse_command_line(argc, argv, cmdline_options), vm);

	//---------------------------------------------------------------------------------------------
	// from file

	std::ifstream ifs("boost_data\\boost_multiple_sources.cfg");
	if (ifs.is_open())
	{
		boost::program_options::store(boost::program_options::parse_config_file(ifs, config_file_options), vm);
		ifs.close();
	}

	boost::program_options::notify(vm);

	//---------------------------------------------------------------------------------------------
	// step #5

	if (vm.count("help"))
	{
		std::cout << visible << std::endl;
		return true;
	}

	if (vm.count("version"))
	{
		std::cout << "Multiple sources example, version 1.0" << std::endl;
		return true;
	}

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

	return true;
}
