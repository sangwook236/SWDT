#include <iostream>
#include <string>


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
		catch (const std::invalid_argument &e)
		{
			std::cout << "std::invalid_argument occurred" << std::endl;
		}

		try
		{
			const std::string str5 = "12345678901234567890";
			const int myint5 = std::stoi(str5);  // error: std::out_of_range.
			std::cout << "std::stoi(\"" << str5 << "\") is " << myint5 << std::endl;
		}
		catch (const std::out_of_range &e)
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
