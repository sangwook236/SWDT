#include <iostream>
#include <string>
#include <locale>
#include <codecvt>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void string()
{
	// basic.
	{
		const char hello[] = { '\x48', '\x65', '\x6c', '\x6c', '\x6f', '\0' };
		std::cout << hello << std::endl;
	}

	// conversion 1.
	{
		const std::string str1 = "45";
		const int myint1 = std::stoi(str1);
		std::cout << "std::stoi(\"" << str1 << "\") is " << myint1 << std::endl;

		const std::string strOct = "045";
		const int myoct = std::stoi(strOct, nullptr, 8);
		std::cout << "std::stoi(\"" << std::oct << strOct << "\") is 0" << myoct << " (" << std::dec << myoct << ')' << std::endl;

		const std::string strHex = "0x45";
		const int myhex = std::stoi(strHex, nullptr, 16);
		std::cout << "std::stoi(\"" << std::hex << strHex << "\") is 0x" << myhex << " (" << std::dec << myhex << ')' << std::endl;

		const std::string str2 = "3.14159";
		const int myint2 = std::stoi(str2);
		std::cout << "std::stoi(\"" << str2 << "\") is " << myint2 << std::endl;

		const std::string str3 = "31337 with words";
		const int myint3 = std::stoi(str3);
		std::cout << "std::stoi(\"" << str3 << "\") is " << myint3 << std::endl;

		try
		{
			const std::string str4 = "words and 2";
			//const int myint4 = std::stoi(str4);  // error: std::invalid_argument.
			std::size_t pos4 = -1;
			const int myint4 = std::stoi(str4, &pos4);  // error: std::invalid_argument.
			std::cout << "std::stoi(\"" << str4 << "\") is " << myint4 << std::endl;
		}
		catch (const std::invalid_argument &ex)
		{
			std::cout << "std::invalid_argument occurred" << std::endl;
		}

		try
		{
			const std::string str5 = "12345678901234567890";
			const int myint5 = std::stoi(str5);  // error: std::out_of_range.
			std::cout << "std::stoi(\"" << str5 << "\") is " << myint5 << std::endl;
		}
		catch (const std::out_of_range &ex)
		{
			std::cout << "std::out_of_range occurred" << std::endl;
		}
	}

	// conversion 2.
	{
		const std::string stri = "45";
		const std::string strl = "45l";
		const std::string strll = "45ll";
		const std::string strul = "45ul";
		const std::string strull = "45ull";
		const std::string strf = "45.0f";
		const std::string strd = "45.0";
		const std::string strld = "45.0l";

		const int numi = std::stoi(stri);
		const long numl = std::stol(strl);
		const long long numll = std::stoll(strll);
		const unsigned long numul = std::stoul(strul);
		const unsigned long long numull = std::stoull(strull);
		const float numf = std::stof(strf);
		const double numd = std::stod(strd);
		const long double numld = std::stold(strld);

		std::cout << "std::stoi(\"" << stri << "\") is " << numi << std::endl;
		std::cout << "std::stol(\"" << strl << "\") is " << numl << std::endl;
		std::cout << "std::stoll(\"" << strll << "\") is " << numll << std::endl;
		std::cout << "std::stoul(\"" << strul << "\") is " << numul << std::endl;
		std::cout << "std::stoull(\"" << strull << "\") is " << numull << std::endl;
		std::cout << "std::stof(\"" << strf << "\") is " << numf << std::endl;
		std::cout << "std::stod(\"" << strd << "\") is " << numd << std::endl;
		std::cout << "std::stold(\"" << strld << "\") is " << numld << std::endl;
	}
}

void unicode_string()
{
	// Print Korean and Chinese.
	{
		//std::locale::global(std::locale("UTF-8"));
		std::locale::global(std::locale("kor"));

		std::cout << "Set locale to " << std::locale().name() << std::endl;

		//std::wcout.imbue(std::locale("kor"));
		//std::wcin.imbue(std::locale("kor"));

		std::wcout << L"ÇÑ±Û Ãâ·Â Å×½ºÆ®." << std::endl;
		std::wcout << L"ÓÞùÛÚÅÏÐ." << std::endl;
	}

	// UTF-8 <--> wide string.
	{
		// UTF-8 data. The character U+1d10b, musical sign segno, does not fit in UCS2.
		const std::string utf8(u8"z\u6c34\U0001d10b");
		for (const auto &c : utf8)
			std::cout << std::hex << std::showbase << (int)c << std::endl;

		std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

		// Convert UTF-8 string to wstring.
		const std::wstring wstr = conv.from_bytes(utf8);
		std::wcout << "wstr = " << wstr << std::endl;

		// Convert wstring to UTF-8 string.
		const std::string utf8_cvt = conv.to_bytes(wstr);
		for (const auto &c : utf8_cvt)
			std::cout << std::hex << std::showbase << (int)c << std::endl;
	}

#if false
	// UTF-8 <--> UTF-16.
	// REF [site] >> https://en.cppreference.com/w/cpp/locale/codecvt_utf8
	{
		// UTF-8 data. The character U+1d10b, musical sign segno, does not fit in UCS2.
		const std::string utf8(u8"z\u6c34\U0001d10b");

		// the UTF-8 / UTF-16 standard conversion facet.
		std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16conv;

		const std::u16string utf16 = utf16conv.from_bytes(utf8);  // UTF-8 to UTF-16.
		std::cout << "UTF-16 conversion produced " << utf16.size() << " code units:" << std::endl;
		for (char16_t c : utf16)
			std::cout << std::hex << std::showbase << (int)c << std::endl;

		const std::string utf8_cvt = utf16conv.to_bytes(utf16);  // UTF-16 to UTF-8.
		std::cout << "UTF-8 conversion produced " << utf8_cvt.size() << " code units:" << std::endl;
		for (char c : utf8_cvt)
			std::cout << std::hex << std::showbase << (int)c << std::endl;

		// the UTF-8 / UCS2 standard conversion facet.
		std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> ucs2conv;
		try
		{
			std::u16string ucs2 = ucs2conv.from_bytes(utf8);
		}
		catch (const std::range_error &ex)
		{
			const std::u16string ucs2 = ucs2conv.from_bytes(utf8.substr(0, ucs2conv.converted()));
			std::cout << "UCS2 failed after producing " << std::dec << ucs2.size() << " characters:" << std::endl;
			for (char16_t c : ucs2)
				std::cout << std::hex << std::showbase << (int)c << std::endl;
		}
	}
#endif
}

void string_tokenization()
{
	// REF [file] >> boost/tokenizer.cpp.
	throw std::runtime_error("Not yet implemented");
}
