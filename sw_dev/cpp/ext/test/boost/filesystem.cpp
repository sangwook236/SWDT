#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/progress.hpp>
#include <iostream>


namespace {
namespace local {

void path_info_1()
{
	boost::filesystem::path my_path("data/boost/filesystem_test.txt");

	boost::filesystem::remove_all("data/boost/foobar");
	boost::filesystem::create_directory("data/boost/foobar");

	// create file
	boost::filesystem::ofstream file("data/boost/foobar/cheeze.wine");
	file << "tastes good!\n";
	file.close();

	if (!boost::filesystem::exists("data/boost/foobar/cheeze.wine"))
		std::cout << "Something is rotten in foobar" << std::endl;

	std::cout << "[ 1] " << my_path.string() << std::endl;
	//std::cout << "[ 2] " << my_path.file_string() << std::endl;  // deprecated
	//std::cout << "[ 3] " << my_path.directory_string() << std::endl;  // deprecated

	//std::cout << "[ 4] " << my_path.external_file_string() << std::endl;  // deprecated
	///std::cout << "[ 5] " << my_path.external_directory_string() << std::endl;  // deprecated
#if !defined(__GNUC__)
	std::wcout << L"[ 4] " << my_path.native() << std::endl;
#endif

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

const char * say_what(bool b)  { return b ? "true" : "false"; }

void path_info_2(const std::string &path)
{
/*
	if (argc < 2)
	{
		std::cout << "Usage: path_info path-portion...\n"
			"Example: path_info foo/bar baz\n"
#ifdef BOOST_POSIX_API
			"         would report info about the composed path foo/bar/baz\n";
#else  // BOOST_WINDOWS_API
			"         would report info about the composed path foo/bar\\baz\n";
#endif
		return 1;
	}

	boost::filesystem::path p;  //  compose a path from the command line arguments

	for (; argc > 1; --argc, ++argv)
		p /= argv[1];
*/

	boost::filesystem::path p(path.c_str());  // compose a path from the command line arguments

	std::cout  <<  "\ncomposed path:" << std::endl;
	std::cout  <<  "  std::cout << -------------: " << p << std::endl;
	std::cout  <<  "  make_preferred()----------: " << boost::filesystem::path(p).make_preferred() << std::endl;

	std::cout << "\nelements:" << std::endl;

	for (boost::filesystem::path::iterator it(p.begin()), it_end(p.end()); it != it_end; ++it)
		std::cout << "  " << *it << std::endl;

	std::cout  <<  "\nobservers, native format:" << std::endl;
# ifdef BOOST_POSIX_API
	std::cout  <<  "  native()-------------: " << p.native() << std::endl;
	std::cout  <<  "  c_str()--------------: " << p.c_str() << std::endl;
# else  // BOOST_WINDOWS_API
	std::wcout << L"  native()-------------: " << p.native() << std::endl;
	std::wcout << L"  c_str()--------------: " << p.c_str() << std::endl;
# endif
	std::cout  <<  "  string()-------------: " << p.string() << std::endl;
	std::wcout << L"  wstring()------------: " << p.wstring() << std::endl;

	std::cout  <<  "\nobservers, generic format:" << std::endl;
	std::cout  <<  "  generic_string()-----: " << p.generic_string() << std::endl;
	std::wcout << L"  generic_wstring()----: " << p.generic_wstring() << std::endl;

	std::cout  <<  "\ndecomposition:" << std::endl;
	std::cout  <<  "  root_name()----------: " << p.root_name() << std::endl;
	std::cout  <<  "  root_directory()-----: " << p.root_directory() << std::endl;
	std::cout  <<  "  root_path()----------: " << p.root_path() << std::endl;
	std::cout  <<  "  relative_path()------: " << p.relative_path() << std::endl;
	std::cout  <<  "  parent_path()--------: " << p.parent_path() << std::endl;
	std::cout  <<  "  filename()-----------: " << p.filename() << std::endl;
	std::cout  <<  "  stem()---------------: " << p.stem() << std::endl;
	std::cout  <<  "  extension()----------: " << p.extension() << std::endl;

	std::cout  <<  "\nquery:" << std::endl;
	std::cout  <<  "  empty()--------------: " << say_what(p.empty()) << std::endl;
	std::cout  <<  "  is_absolute()--------: " << say_what(p.is_absolute()) << std::endl;
	std::cout  <<  "  has_root_name()------: " << say_what(p.has_root_name()) << std::endl;
	std::cout  <<  "  has_root_directory()-: " << say_what(p.has_root_directory()) << std::endl;
	std::cout  <<  "  has_root_path()------: " << say_what(p.has_root_path()) << std::endl;
	std::cout  <<  "  has_relative_path()--: " << say_what(p.has_relative_path()) << std::endl;
	std::cout  <<  "  has_parent_path()----: " << say_what(p.has_parent_path()) << std::endl;
	std::cout  <<  "  has_filename()-------: " << say_what(p.has_filename()) << std::endl;
	std::cout  <<  "  has_stem()-----------: " << say_what(p.has_stem()) << std::endl;
	std::cout  <<  "  has_extension()------: " << say_what(p.has_extension()) << std::endl;
}

bool find_file(const boost::filesystem::path &dir_path, const std::string &file_name, boost::filesystem::path &path_found)
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
#if defined(__GNUC__)
	boost::filesystem::path p(path.c_str());
#else
	boost::filesystem::path p(path.c_str(), boost::filesystem::native);
#endif

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

#if defined(__GNUC__)
	full_path = boost::filesystem::system_complete(boost::filesystem::path(path.c_str()));
#else
	full_path = boost::filesystem::system_complete(boost::filesystem::path(path.c_str(), boost::filesystem::native));
#endif

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
		std::cout << "\nIn directory: " << full_path.string() << std::endl << std::endl;
		boost::filesystem::directory_iterator end_iter;
		for (boost::filesystem::directory_iterator dir_itr(full_path); dir_itr != end_iter; ++dir_itr)
		{
			try
			{
				if (boost::filesystem::is_directory(dir_itr->status()))
				{
					++dir_count;
					std::cout << dir_itr->path().filename() << " [directory]" << std::endl;
				}
				else if (boost::filesystem::is_regular_file(dir_itr->status()))
				{
					++file_count;
					std::cout << dir_itr->path().filename() << std::endl;
				}
				else
				{
					++other_count;
					std::cout << dir_itr->path().filename() << " [other]" << std::endl;
				}

			}
			catch (const std::exception &ex)
			{
				++err_count;
				std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
			}
		}

		std::cout << "\n" << file_count << " files" << std::endl
			<< dir_count << " directories" << std::endl
			<< other_count << " others" << std::endl
			<< err_count << " errors" << std::endl;
	}
	else // must be a file
	{
		std::cout << "\nFound: " << full_path.string() << std::endl;
	}

	return true;
}

bool rename_files_in_directory(const boost::filesystem::path &dir_path)
{
	if (!boost::filesystem::exists(dir_path)) return false;

	boost::filesystem::directory_iterator end_itr;  // default construction yields past-the-end
	for (boost::filesystem::directory_iterator itr(dir_path); itr != end_itr; ++itr)
	{
		if (boost::filesystem::is_regular_file(itr->status()) && itr->path().extension() == ".ppm")
		{
			const std::size_t pos = itr->path().string().find_last_of("in");
			if (pos != std::string::npos)
			{
				const std::string fn(itr->path().string().substr(0, pos+1) + ".1" + itr->path().string().substr(pos+1));
				boost::filesystem::rename(itr->path(), boost::filesystem::path(fn));
			}
		}
	}

	return true;
}

}  // namespace local
}  // unnamed namespace

void filesystem()
{
	//local::path_info_1();
	//local::path_info_2("./data/boost/foobar");
	//local::path_info_2("./data/boost/foobar/cheeze.wine");
	
	boost::filesystem::path path_found(boost::filesystem::initial_path<boost::filesystem::path>());
	if (local::find_file(".", "cheeze.wine", path_found))
		std::cout << "File Found: " << path_found.string() << std::endl;
	else
		std::cout << "File Not Found" << std::endl;

	//local::file_size("data/boost/foobar/cheeze.wine");
	//local::ls("data/boost/foobar/");

	//local::rename_files_in_directory("E:/archive_dataset/change_detection/canoe_ppm");
	//local::rename_files_in_directory("E:/archive_dataset/change_detection/highway_ppm");
	local::rename_files_in_directory("E:/archive_dataset/change_detection/boats_ppm");
}
