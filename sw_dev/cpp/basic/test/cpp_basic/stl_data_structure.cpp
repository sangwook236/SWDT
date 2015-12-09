#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <string>


namespace {
namespace local {

void unordered_set()
{
	std::unordered_set<std::string> myset = { "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune" };

	const unsigned n = myset.bucket_count();

	std::cout << "size = " << myset.size() << std::endl;
	std::cout << "bucket_count = " << myset.bucket_count() << std::endl;
	std::cout << "load_factor = " << myset.load_factor() << std::endl;
	std::cout << "max_load_factor = " << myset.max_load_factor() << std::endl;

	for (unsigned i = 0; i < n; ++i)
	{
		std::cout << "bucket #" << i << " has " << myset.bucket_size(i) << " elements." << std::endl;
		std::cout << "bucket #" << i << " contains:";
		for (auto it = myset.begin(i); it != myset.end(i); ++it)
			std::cout << " " << (*it);
		std::cout << std::endl;
	}

	//
	for (const std::string &x : myset)
	{
		std::cout << x << " is in bucket #" << myset.bucket(x) << std::endl;
	}

	//
	std::unordered_set<std::string>::const_iterator got = myset.find("Earth");
	if (myset.end() == got)
		std::cout << "not found in myset" << std::endl;
	else
		std::cout << *got << " is in myset" << std::endl;
}

void unordered_map()
{
	std::unordered_map<std::string, std::string> mymap = {
		{ "us", "United States" },
		{ "uk", "United Kingdom" },
		{ "fr", "France" },
		{ "de", "Germany" },
		{ "kr", "Korea" }
	};

	const unsigned nbuckets = mymap.bucket_count();

	std::cout << "size = " << mymap.size() << std::endl;
	std::cout << "bucket_count = " << mymap.bucket_count() << std::endl;
	std::cout << "load_factor = " << mymap.load_factor() << std::endl;
	std::cout << "max_load_factor = " << mymap.max_load_factor() << std::endl;

	for (unsigned i = 0; i < nbuckets; ++i)
	{
		std::cout << "bucket #" << i << " has " << mymap.bucket_size(i) << " elements." << std::endl;
	}

	for (auto &x : mymap)
	{
		std::cout << "Element [" << x.first << ":" << x.second << "]";
		std::cout << " is in bucket #" << mymap.bucket(x.first) << std::endl;
	}

	//
	std::cout << "mymap contains:";
	for (auto it = mymap.begin(); it != mymap.end(); ++it)
		std::cout << " " << it->first << ":" << it->second;
	std::cout << std::endl;

	//
	std::cout << "mymap's buckets contain:" << std::endl;
	for (unsigned i = 0; i < mymap.bucket_count(); ++i)
	{
		std::cout << "bucket #" << i << " contains:";
		for (auto local_it = mymap.begin(i); local_it != mymap.end(i); ++local_it)
			std::cout << " " << local_it->first << ":" << local_it->second;
		std::cout << std::endl;
	}

	//
	std::unordered_map<std::string, std::string>::const_iterator got = mymap.find("kr");
	if (mymap.end() == got)
		std::cout << "not found" << std::endl;
	else
		std::cout << got->first << " is " << got->second << std::endl;
}

}  // namespace local
}  // unnamed namespace

void stl_data_structure()
{
	//local::unordered_set();
	local::unordered_map();
}
