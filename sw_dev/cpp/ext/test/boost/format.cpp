#include <boost/format.hpp>
#include <iostream>
#include <string>


namespace {
namespace local {

void basic()
{
	std::cout << boost::format("%2% %1%") % 36 % 77 << std::endl;

	boost::format fmter("%2% %1%");
	fmter % 36; fmter % 77;
	std::cout << fmter << std::endl;

	std::string s = fmter.str();
	// Possibly several times:
	s = fmter.str();
	std::cout << s << std::endl;

	// Using the str free function:
	const std::string s2 = boost::str(boost::format("%2% %1%") % 36 % 77);
	std::cout << s2 << std::endl;
}

}  // namespace local
}  // unnamed namespace

void format()
{
	local::basic();
}
