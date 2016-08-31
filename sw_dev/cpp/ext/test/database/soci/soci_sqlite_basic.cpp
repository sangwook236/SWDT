//#include "stdafx.h"
#include <soci/sqlite3/soci-sqlite3.h>
#include <soci/soci.h>
#include <boost/fusion/include/define_struct.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <string>
#include <iostream>


BOOST_FUSION_DEFINE_STRUCT(
(my_soci), AccountTable,
(int, Id)
(std::string, Account)
(short, Grade)
)

namespace {
namespace local {

struct AccountStruct
{
	int Id;
	std::string Account;
	short Grade;
};

void simple_example_1()
{
	soci::session sql(soci::sqlite3, "./data/database/simple.sqlite");

#if 0
	my_soci::AccountTable accounts;
	sql << "select * from tbl_Account", soci::into(accounts);  // NOTICE [error] >> Compile-time error.
	boost::fusion::for_each(accounts, [](auto const& e) { std::cout << e << ' '; });
#elif 0
	std::vector<AccountStruct> accounts;
	sql << "SELECT * FROM tbl_Account", soci::into(accounts);  // NOTICE [error] >> Compile-time error.
	for (auto account : accounts)
		std::cout << account.Id << '\t' << account.Account << '\t' << account.Grade << std::endl;
#endif
	std::cout << std::endl;
}

void simple_example_2()
{
	//soci::session sql(soci::sqlite3, "./data/database/test.sqlite");  // File DB.
	soci::session sql(soci::sqlite3, ":memory:");  // Memory DB.

	sql << "CREATE TABLE IF NOT EXISTS tbl (i INTEGER, s TEXT)";
	sql << "INSERT INTO tbl (i, s) VALUES (0, 'first')";
	sql << "INSERT INTO tbl (i, s) VALUES (1, 'second')";
	sql << "INSERT INTO tbl (i, s) VALUES (2, 'third')";

	int num;
	std::string str;
	sql << "SELECT i, s FROM tbl WHERE i = 0", soci::into(num), soci::into(str);
	std::cout << num << '\t' << str << std::endl;

	//boost::tuple<int, std::string> r;
	//sql << "SELECT i, s FROM tbl WHERE i = 0", soci::into(r);  // NOTICE [error] >> Compile-time error.
	//std::cout << r.get<0>() << '\t' << r.get<1>() << std::endl;

}

}  // namespace local
}  // unnamed namespace

namespace my_soci {

void sqlite_basic()
{
	try
	{
		local::simple_example_1();
		local::simple_example_2();
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
	}
}

}  // namespace my_soci
