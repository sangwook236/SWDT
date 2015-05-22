//#include "stdafx.h"
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <iterator>


namespace {
namespace local {

void method1()
{
	const std::string str("Splitting   a string in C++//");
    std::istringstream sstream(str);

	// use stream iterators to copy the stream to the vector as whitespace separated strings.
#if 1
	std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(sstream), std::istream_iterator<std::string>(), std::back_inserter(tokens));
#else
	const std::vector<std::string> tokens(std::istream_iterator<std::string>(sstream), std::istream_iterator<std::string>());
#endif

	// display.
	std::copy(tokens.begin(), tokens.end(), std::ostream_iterator<std::string>(std::cout, "."));
	std::cout << std::endl;
}

void tokenize_string(const std::string &str, std::vector<std::string> &tokens, const char delimiter)
{
	std::istringstream sstream(str);

	std::string tok;
	while (std::getline(sstream, tok, delimiter))
		tokens.push_back(tok);
}

void method2()
{
	const std::string str("Hello,How Are , You,Today");

	std::vector<std::string> tokens;
	tokenize_string(str, tokens, ',');

	// display.
	std::copy(tokens.begin(), tokens.end(), std::ostream_iterator<std::string>(std::cout, "."));
	std::cout << std::endl;
}

void method3()
{
	// use Boost.Tokenizer.

    const std::string str("token, test   string");

    boost::char_separator<char> sep(", ");
    boost::tokenizer<boost::char_separator<char> > tokens(str, sep);

	// display.
	BOOST_FOREACH(const std::string &tok, tokens)
        std::cout << tok << '.';
	std::cout << std::endl;
}

void method4()
{
	// use Boost.StringAlgorithms.

	const std::string str("abc,bcd, cde ,,efg,fgh,");

	std::vector<std::string> tokens;
	//boost::split(tokens, str, boost::is_any_of(","));
	boost::split(tokens, str, boost::is_any_of(", "));

	// display.
	BOOST_FOREACH(const std::string &tok, tokens)
	//for (const auto &tok : tokens)
		std::cout << tok << '.';
	std::cout << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_string_tokenization {

}  // namespace my_string_tokenization

int string_tokenization_main(int argc, char *argv[])
{
	local::method1();
	local::method2();
	local::method3();
	local::method4();

	return 0;
}
