#include <boost/tokenizer.hpp>
#include <string>
#include <iostream>


void tokenizer()
{
#if defined(UNICODE) || defined(_UNICODE)
	typedef boost::tokenizer<boost::char_separator<wchar_t>, std::wstring::const_iterator, std::wstring> tokenizer_type;
#else
	typedef boost::tokenizer<> tokenizer_type;
#endif

#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring str = L";;Hello|world||-foo--bar;yow;baz|";
#else
	const std::string str = ";;Hello|world||-foo--bar;yow;baz|";
#endif

	{
#if defined(UNICODE) || defined(_UNICODE)
		const boost::char_separator<wchar_t> sep(L"-;|");
#else
		const boost::char_separator<char> sep("-;|");
#endif
		tokenizer_type tokens(str, sep);

		for (tokenizer_type::iterator it = tokens.begin(); it != tokens.end(); ++it)
#if defined(UNICODE) || defined(_UNICODE)
			std::wcout << L'<' << *it << L"> ";
		std::wcout << std::endl;
#else
			std::cout << '<' << *it << "> ";
		std::cout << std::endl;
#endif
	}

	{
#if defined(UNICODE) || defined(_UNICODE)
		const boost::char_separator<wchar_t> sep(L"-;", L"|", boost::keep_empty_tokens);
#else
		const boost::char_separator<char> sep("-;", "|", boost::keep_empty_tokens);
#endif

		tokenizer_type tokens(str, sep);

		for (tokenizer_type::iterator it = tokens.begin(); it != tokens.end(); ++it)
#if defined(UNICODE) || defined(_UNICODE)
			std::wcout << L'<' << *it << L"> ";
		std::wcout << std::endl;
#else
			std::cout << '<' << *it << "> ";
		std::cout << std::endl;
#endif
	}

#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring str2 = L"This is,  a test.";
#else
	const std::string str2 = "This is,  a test.";
#endif

	{
#if defined(UNICODE) || defined(_UNICODE)
		const boost::char_separator<wchar_t> sep;
#else
		const boost::char_separator<char> sep;
#endif

		tokenizer_type tokens(str2, sep);

		for (tokenizer_type::iterator it = tokens.begin();	it != tokens.end(); ++it)
#if defined(UNICODE) || defined(_UNICODE)
			std::wcout << L'<' << *it << L"> ";
		std::wcout << std::endl;
#else
			std::cout << '<' << *it << "> ";
		std::cout << std::endl;
#endif
	}
}
