//#include "stdafx.h"
#include <soci/sqlite3/soci-sqlite3.h>
#include <soci/boost-tuple.h>
#include <soci/boost-fusion.h>
#include <soci/soci.h>
#include <boost/fusion/include/define_struct.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <string>
#include <iostream>


BOOST_FUSION_DEFINE_STRUCT(
(my_soci), FruitTable,
(int, id)
(std::string, name)
//(float, price)  // NOTICE [error] >> Compile-time error: 'x_type' undeclared identifier. REF [file] >> enum exchange_type in ${SOCI_HOME}/include/soci/soci-backend.h.
(double, price)
)

BOOST_FUSION_DEFINE_STRUCT(
(my_soci), AccountTable,
(int, Id)
(std::string, Account)
(short, Grade)
)

namespace {
namespace local {

struct FruitRow
{
	int id;
	std::string name;
	float price;
};

struct AccountRow
{
	int Id;
	std::string Account;
	short Grade;
};

void simple_example_1()
{
	//
	{
		soci::session sql(soci::sqlite3, "./data/database/fruit.sqlite3");

		//
		{
			my_soci::FruitTable fruit;
			sql << "select * from Fruit", soci::into(fruit);
			boost::fusion::for_each(fruit, [](auto const& elem) { std::cout << elem << " : "; });
			std::cout << std::endl;

			// Not correctly working: boost::fusion::vector is different from std::vector, rather is similar to boost::tuple.
			//boost::fusion::vector<my_soci::FruitTable> fruits;
			//// Stupid initialization.
			//for (int i = 0; i < 3; ++i)
			//	boost::fusion::push_back(fruits, my_soci::FruitTable());
			//std::cout << "size = " << boost::fusion::size(fruits) << std::endl;
			//sql << "select * from Fruit", soci::into(fruits);
			//boost::fusion::for_each(fruits, [](auto const& elem) { std::cout << elem.id << " : " << elem.name << " : " << elem.price; });
			//std::cout << std::endl;
		}

		//
#if 0
		{
			FruitRow fruit;
			//sql << "SELECT * FROM Fruit", soci::into(fruit);  // NOTICE [error] >> Compile-time error: 'x_type' undeclared identifier.
			std::cout << fruit.id << " : " << fruit.name << " : " << fruit.price << std::endl;
			std::cout << std::endl;

			std::vector<FruitRow> fruits(3);
			//sql << "SELECT * FROM Fruit", soci::into(fruits);  // NOTICE [error] >> Compile-time error: 'x_type' undeclared identifier.
			for (auto elem : fruits)
				std::cout << elem.id << " : " << elem.name << " : " << elem.price;
			std::cout << std::endl;

			// Not correctly working: boost::fusion::vector is different from std::vector, rather is similar to boost::tuple.
			//boost::fusion::vector<FruitRow> fruits2(3);  // NOTICE [error] >> Compile-time error.
			//sql << "SELECT * FROM Fruit", soci::into(fruits2);
			//boost::fusion::for_each(fruits2, [](auto const& elem) { std::cout << elem.id << " : " << elem.name << " : " << elem.price; });
			//std::cout << std::endl;
		}
#endif

		//
		std::vector<double> prices(3);
		sql << "select price from Fruit", soci::into(prices);
		for (auto price : prices)
			std::cout << price << ", ";
		std::cout << std::endl;
	}

	//
	{
		soci::session sql(soci::sqlite3, "./data/database/simple.sqlite3");

		{
			my_soci::AccountTable accounts;
			sql << "select * from tbl_Account", soci::into(accounts);
			boost::fusion::for_each(accounts, [](auto const& elem) { std::cout << elem << " : "; });
			std::cout << std::endl;
		}

#if 0
		{
			std::vector<AccountRow> accounts(5);
			sql << "SELECT * FROM tbl_Account", soci::into(accounts);  // NOTICE [error] >> Compile-time error: 'x_type' undeclared identifier.
			for (auto account : accounts)
				std::cout << account.Id << " : " << account.Account << " : " << account.Grade << std::endl;
			std::cout << std::endl;
		}
#endif
	}
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

	boost::tuple<int, std::string> r;
	sql << "SELECT i, s FROM tbl WHERE i = 0", soci::into(r);
	std::cout << r.get<0>() << '\t' << r.get<1>() << std::endl;

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
