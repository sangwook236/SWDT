#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <iostream>
#include <string>
#include <vector>


namespace {
namespace local {

// REF [site] >> http://www.boost.org/doc/libs/1_58_0/doc/html/string_algo/usage.html
void conversion_example()
{
	std::string str1(" hello world! ");
	boost::to_upper(str1);  // str1 == " HELLO WORLD! ".
	std::cout << str1 << std::endl;
	boost::trim(str1);  // str1 == "HELLO WORLD!".
	std::cout << str1 << std::endl;

	const std::string str2 = boost::to_lower_copy(boost::ireplace_first_copy(str1, "hello", "goodbye"));  // str2 == "goodbye world!".
	std::cout << str2 << std::endl;
}

bool is_executable(const std::string &filename)
{
	return boost::iends_with(filename, ".exe") || boost::iends_with(filename, ".com");
}

void predicates_and_classification_example()
{
    // ...
	const std::string str1("command.com");
    std::cout 
        << str1
        << (is_executable(str1) ? "is" : "is not") 
        << "an executable" 
        << std::endl;  // Print "command.com is an executable".
    
    //
	const char text1[] = "hello";
    std::cout 
        << text1 
        << (boost::all(text1, boost::is_lower()) ? " is" : " is not")
        << " written in the lower case" 
        << std::endl;  // Print "hello is written in the lower case".
}

void trimming_example()
{
	std::string str1("     hello world!     ");
	boost::trim(str1);  // str1 == "hello world!"
	std::cout << str1 << std::endl;
	const std::string str2(boost::trim_left_copy(str1));  // str2 == "hello world!     "
	std::cout << str1 << std::endl;
	const std::string str3(boost::trim_right_copy(str1));  // str3 == "     hello world!"
	std::cout << str1 << std::endl;

	std::string phone("00423333444");
	// Remove leading 0 from the phone number.
	boost::trim_left_if(phone, boost::is_any_of("0"));  // phone == "423333444"
	std::cout << phone << std::endl;
}

void find_example()
{
	char text[] = "hello dolly!";
	boost::iterator_range<char *> result = boost::find_last(text, "ll");

	std::transform(result.begin(), result.end(), result.begin(), std::bind2nd(std::plus<char>(), 1));
	//text = "hello dommy!"
	std::cout << text << std::endl;

	boost::to_upper(result);  // text == "hello doMMy!"
	std::cout << text << std::endl;

	// iterator_range is convertible to bool.
	if (boost::find_first(text, "dolly"))
		std::cout << "Dolly is there" << std::endl;
}

void replace_example()
{
	std::string str1("Hello  Dolly,   Hello World!");
	std::cout << str1 << std::endl;
	boost::replace_first(str1, "Dolly", "Jane");  // str1 == "Hello  Jane,   Hello World!"
	std::cout << str1 << std::endl;
	boost::replace_last(str1, "Hello", "Goodbye");  // str1 == "Hello  Jane,   Goodbye World!"
	std::cout << str1 << std::endl;
	boost::erase_all(str1, " ");  // str1 == "HelloJane,GoodbyeWorld!"
	std::cout << str1 << std::endl;
	boost::erase_head(str1, 6);  // str1 == "Jane,GoodbyeWorld!"
	std::cout << str1 << std::endl;
}

void find_iterator()
{
	// FIXME [error] >> boost::make_find_iterator not found => don't know why.
/*
	const std::string str1("abc-*-ABC-*-aBc");
	// Find all 'abc' substd::strings (ignoring the case).
	// Create a find_iterator.
	typedef boost::find_iterator<std::string::iterator> string_find_iterator;
	for (string_find_iterator it = boost::make_find_iterator(str1, boost::first_finder("abc", boost::is_iequal())); it != string_find_iterator(); ++it)
	{
		std::cout << boost::copy_range<std::string>(*it) << std::endl;

		// Shift all chars in the match by one.
        std::transform(it->begin(), it->end(), it->begin(), std::bind2nd(std::plus<char>(), 1));
	}

	// Output will be:
	// abc
	// ABC
	// aBC

	typedef boost::split_iterator<std::string::iterator> string_split_iterator;
	for (string_split_iterator it =	boost::make_split_iterator(str1, boost::first_finder("-*-", boost::is_iequal())); it != string_split_iterator(); ++it)
		std::cout << boost::copy_range<std::string>(*it) << std::endl;

	// Output will be:
	// abc
	// ABC
	// aBC
*/
}

void split_example()
{
	std::string str1("hello abc-*-ABC-*-aBc goodbye");

	//
	typedef std::vector<boost::iterator_range<std::string::iterator> > find_vector_type;

	find_vector_type findVec;  // #1: search for separators
	boost::ifind_all(findVec, str1, "abc");  // findVec == { [abc],[ABC],[aBc] }

	for (find_vector_type::iterator it = findVec.begin(); it != findVec.end(); ++it)
		std::cout << std::string(it->begin(), it->end()) << std::endl;

	//
	typedef std::vector<std::string> split_vector_type;

	split_vector_type splitVec;  // #2: search for tokens
	boost::split(splitVec, str1, boost::is_any_of("-*-"), boost::token_compress_on);  // splitVec == { "hello abc","ABC","aBc goodbye" }

	for (split_vector_type::iterator it = splitVec.begin(); it != splitVec.end(); ++it)
		std::cout << std::string(it->begin(), it->end()) << std::endl;
}

void string_tokenization()
{
	const std::string str("a,b, c ,,e,f,");

	std::vector<std::string> tokens;
	boost::split(tokens, str, boost::is_any_of(","));

	// Output.
	BOOST_FOREACH(const std::string &tok, tokens)
	//for (const auto &tok : tokens)
		std::cout << tok << '.';
	std::cout << std::endl;
}

}  // namespace local
}  // unnamed namespace

void string_algorithms()
{
	local::conversion_example();
	local::predicates_and_classification_example();
	local::trimming_example();
	local::find_example();
	local::replace_example();
	local::find_iterator();
	local::split_example();

	local::string_tokenization();
}
