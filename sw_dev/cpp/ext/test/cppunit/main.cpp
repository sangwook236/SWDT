#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <iostream>


int main(int argc, char *argv[])
{
	try
	{
		CppUnit::TextUi::TestRunner runner;
		runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

		runner.run();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred: " << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}
