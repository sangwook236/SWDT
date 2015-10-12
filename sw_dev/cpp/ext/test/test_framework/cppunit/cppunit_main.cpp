#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_cppunit {

}  // namespace my_cppunit

int cppunit_main(int argc, char *argv[])
{
	CppUnit::TextUi::TestRunner runner;
	runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

	return runner.run() ? EXIT_SUCCESS : EXIT_FAILURE;
}
