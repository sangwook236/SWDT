//#include "stdafx.h"
#if __cplusplus >= 201103L
#include "lmdb++.h"
#endif
#include <iostream>


namespace {
namespace local {

// [ref] ${LMDBXX_HOME}/example.cc
void example()
{
#if __cplusplus >= 201103L
	// Create and open the LMDB environment.
	auto env = lmdb::env::create();
	env.open("./example.mdb", 0, 0664);

	// Insert some key/value pairs in a write transaction.
	auto wtxn = lmdb::txn::begin(env);
	auto dbi = lmdb::dbi::open(wtxn, nullptr);
	dbi.put(wtxn, "username", "jhacker");
	dbi.put(wtxn, "email", "jhacker@example.org");
	dbi.put(wtxn, "fullname", "J. Random Hacker");
	wtxn.commit();

	// Fetch key/value pairs in a read-only transaction.
	auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
	auto cursor = lmdb::cursor::open(rtxn, dbi);
	std::string key, value;
	while (cursor.get(key, value, MDB_NEXT))
	{
		std::printf("key: '%s', value: '%s'\n", key.c_str(), value.c_str());
	}
	cursor.close();
	rtxn.abort();

	// The enviroment is closed automatically.
#else
	std::cerr << "A C++11 compiler (CXXFLAGS='-std=c++11') is required" << std::endl;
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_lmdbxx {

}  // namespace my_lmdbxx

int lmdbxx_main(int argc, char *argv[])
{
	local::example();

	return 0;
}