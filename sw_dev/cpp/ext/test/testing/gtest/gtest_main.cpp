#include <gtest/gtest.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gtest {

}  // namespace my_gtest

int gtest_main(int argc, char *argv[])
{
	// Usage:
	//	REF [site] >> http://www.ibm.com/developerworks/aix/library/au-googletestingframework.html
	//
	//	Command line:
	//		my_gtest_exe --gtest_output="xml:gtest_report.xml" --gtest_repeat=2 --gtest_break_on_failure -gtest_filter=<test string>
	//			The format for <test string> is a series of wildcard patterns separated by colons (:).
	//				--gtest_filter=* : runs all tests
	//				--gtest_filter=SquareRoot* : runs tests which start with SquareRoot.
	//				--gtest_filter=SquareRootTest.*-SquareRootTest.Zero*
	//					SquareRootTest.* means all tests belonging to SquareRootTest.
	//					-SquareRootTest.Zero* means don't run those tests whose names begin with Zero.
	//
	//	Source code:
	//		testing::GTEST_FLAG(output) = "xml:gtest_report.xml";
	//		testing::GTEST_FLAG(filter) = "Test_Cases1*";
	//		testing::GTEST_FLAG(repeat) = 2;
	//
	//	Environment variables:
	//		export GTEST_FILTER="Test_Cases1*"
	//
	//	e.g.)
	//		my_gtest_exe --gtest_list_tests
	//		my_gtest_exe ./s --number-of-input=5 --gtest_filter=Test_Cases1*

	//testing::GTEST_FLAG(output) = "xml:gtest_report.xml";
	//testing::GTEST_FLAG(throw_on_failure) = true;
	//GTEST_FLAG_SET(output, "xml:gtest_report.xml");
	//GTEST_FLAG_SET(throw_on_failure, true);
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
