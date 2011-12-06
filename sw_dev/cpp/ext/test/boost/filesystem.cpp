#include "stdafx.h"
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/progress.hpp>
#include <iostream>


namespace {

bool find_file(const boost::filesystem::path & dir_path, const std::string & file_name, boost::filesystem::path & path_found)
{
	if (!boost::filesystem::exists(dir_path)) return false;

	boost::filesystem::directory_iterator end_itr;  // default construction yields past-the-end
	for (boost::filesystem::directory_iterator itr(dir_path); itr != end_itr; ++itr)
	{
		if (boost::filesystem::is_directory(itr->status()))
		{
			if (find_file(itr->path(), file_name, path_found)) return true;
		}
		else if (itr->path().filename() == file_name)  // see below
		{
			path_found = itr->path();
			return true;
		}
	}

	return false;
}

bool file_size(const std::string &path)
{
	boost::filesystem::path p(path.c_str(), boost::filesystem::native);

	if (!boost::filesystem::exists(p))
	{
		std::cout << "not found: " << path << std::endl;
		return false;
	}

	if (!boost::filesystem::is_regular(p))
	{
		std::cout << "not a regular file: " << path << std::endl;
		return false;
	}

	std::cout << "size of " << path << " is " << boost::filesystem::file_size(p) << std::endl;
	return true;
}

bool ls(const std::string &path)
{
	boost::progress_timer t(std::clog);

	boost::filesystem::path full_path(boost::filesystem::initial_path<boost::filesystem::path>());

	full_path = boost::filesystem::system_complete(boost::filesystem::path(path.c_str(), boost::filesystem::native));

	unsigned long file_count = 0;
	unsigned long dir_count = 0;
	unsigned long other_count = 0;
	unsigned long err_count = 0;

	if (!boost::filesystem::exists(full_path))
	{
		std::cout << "\nNot found: " << full_path.string() << std::endl;
		return false;
	}

	if (boost::filesystem::is_directory(full_path))
	{
		std::cout << "\nIn directory: " << full_path.string() << "\n\n";
		boost::filesystem::directory_iterator end_iter;
		for (boost::filesystem::directory_iterator dir_itr(full_path); dir_itr != end_iter; ++dir_itr)
		{
			try
			{
				if (boost::filesystem::is_directory(dir_itr->status()))
				{
					++dir_count;
					std::cout << dir_itr->path().filename() << " [directory]\n";
				}
				else if (boost::filesystem::is_regular_file(dir_itr->status()))
				{
					++file_count;
					std::cout << dir_itr->path().filename() << "\n";
				}
				else
				{
					++other_count;
					std::cout << dir_itr->path().filename() << " [other]\n";
				}

			}
			catch (const std::exception & ex)
			{
				++err_count;
				std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
			}
		}

		std::cout << "\n" << file_count << " files\n"
			<< dir_count << " directories\n"
			<< other_count << " others\n"
			<< err_count << " errors\n";
	}
	else // must be a file
	{
		std::cout << "\nFound: " << full_path.string() << "\n";    
	}

	return true;
}

}  // unnamed namespace

void filesystem()
{
	{
		boost::filesystem::path my_path("boost_data/filesystem_test.txt");

		boost::filesystem::remove_all("boost_data/foobar");
		boost::filesystem::create_directory("boost_data/foobar");
		boost::filesystem::ofstream file("boost_data/foobar/cheeze");
		file << "tastes good!\n";
		file.close();
		if (!boost::filesystem::exists("boost_data/foobar/cheeze"))
			std::cout << "Something is rotten in foobar" << std::endl;

		std::cout << "[ 1] " << my_path.string() << std::endl;
		//std::cout << "[ 2] " << my_path.file_string() << std::endl;  // deprecated
		//std::cout << "[ 3] " << my_path.directory_string() << std::endl;  // deprecated

		//std::cout << "[ 4] " << my_path.external_file_string() << std::endl;  // deprecated
		///std::cout << "[ 5] " << my_path.external_directory_string() << std::endl;  // deprecated
		std::wcout << L"[ 4] " << my_path.native() << std::endl;

		std::cout << "[ 6] " << my_path.root_name() << std::endl;
		std::cout << "[ 7] " << my_path.root_directory() << std::endl;
		std::cout << "[ 8] " << my_path.root_path() << std::endl;
		std::cout << "[ 9] " << my_path.relative_path() << std::endl;

		std::cout << "[10] " << my_path.parent_path() << std::endl;
		std::cout << "[11] " << my_path.filename() << std::endl;

		std::cout << "[12] " << my_path.stem() << std::endl;
		std::cout << "[13] " << my_path.extension() << std::endl;

		std::cout << "[14] " << std::boolalpha << my_path.empty() << std::endl;
		std::cout << "[15] " << std::boolalpha << my_path.is_complete() << std::endl;
		std::cout << "[16] " << std::boolalpha << my_path.has_root_name() << std::endl;
		std::cout << "[17] " << std::boolalpha << my_path.has_root_directory() << std::endl;
		std::cout << "[18] " << std::boolalpha << my_path.has_root_path() << std::endl;
		std::cout << "[19] " << std::boolalpha << my_path.has_relative_path() << std::endl;
		std::cout << "[20] " << std::boolalpha << my_path.has_filename() << std::endl;
		std::cout << "[21] " << std::boolalpha << my_path.has_parent_path() << std::endl;
	}

	boost::filesystem::path path_found(boost::filesystem::initial_path<boost::filesystem::path>());
	if (find_file(".", "cheeze", path_found))
		std::cout << "File Found: " << path_found.string() << std::endl;
	else
		std::cout << "File Not Found" << std::endl;

	file_size("boost_data/foobar/cheeze");
	ls("boost_data/foobar/");
}
