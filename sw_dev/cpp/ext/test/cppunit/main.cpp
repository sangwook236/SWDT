#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <iostream>


#if defined(_MSC_VER) && defined(_DEBUG)
#include <afx.h>
#define VC_EXTRALEAN  //  Exclude rarely-used stuff from Windows headers
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


int main()
{
	CppUnit::TextUi::TestRunner runner;
	runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

	runner.run();

	std::cout.flush();
	std::cin.get();
	return 0;
}
