//#include "stdafx.h"
#if defined(_WIN64) || defined(_WIN32)
#include <soci/soci-postgresql.h>
#else
#include <soci/postgresql/soci-postgresql.h>
#endif
#include <soci/soci.h>
#include <boost/fusion/include/define_struct.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/tuple/tuple.hpp>
#include <string>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_soci {

void postgresql_basic()
{
	try
	{
		soci::session sql(soci::postgresql, "host=localhost user=postgres password=XXXXXXXX dbname=postgres port=5432");

		sql << "CREATE TABLE IF NOT EXISTS t (i INTEGER, s TEXT)";
		sql << "TRUNCATE TABLE t";
		sql << "INSERT INTO t (i, s) VALUES (0, 'frist')";

		int i;
		std::string s;

		sql << "SELECT i, s FROM t WHERE i = 0", soci::into(i), soci::into(s);
		std::cout << i << "\t" << s << std::endl;

		boost::tuple<int, std::string> r;
		sql << "SELECT i, s FROM t WHERE i = 0", soci::into(r);
		std::cout << r.get<0>() << "\t" << r.get<1>() << std::endl;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
	}
}

}  // namespace my_soci
