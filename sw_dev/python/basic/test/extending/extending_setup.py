# This is an example of a distutils 'setup' script for the example_nt
# sample.  This provides a simpler way of building your extension
# and means you can avoid keeping MSVC solution files etc in source-control.
# It also means it should magically build with all compilers supported by
# python.

# USAGE: you probably want 'setup.py install' - but execute 'setup.py --help'
# for all the details.

# NOTE: This is *not* a sample for distutils - it is just the smallest
# script that can build this.  See distutils docs for more info.

from distutils.core import setup, Extension

simple_extending_mod = Extension(
	'simple_extending',
	sources = ['simple_extending.cpp'],
	libraries = ['python32'],
	include_dirs = ['D:/work_center/sw_dev/python/basic/src/Python-3.2.2/Include'],
	library_dirs = ['D:/work_center/sw_dev/python/basic/src/Python-3.2.2/PCBuild'],
)
greeting_mod = Extension(
	'greeting',
	sources = ['greeting.cpp'],
	libraries = ['python32'],
	include_dirs = ['D:/work_center/sw_dev/python/basic/src/Python-3.2.2/Include'],
	library_dirs = ['D:/work_center/sw_dev/python/basic/src/Python-3.2.2/PCBuild'],
)
greeting_using_boost_mod = Extension(
	'greeting_using_boost',
	sources = ['greeting_using_boost.cpp'],
	libraries = ['python32', 'boost_python3-vc100-mt-1_48'],
	include_dirs = ['D:/work_center/sw_dev/python/basic/src/Python-3.2.2/Include', 'D:/work_center/sw_dev/cpp/ext/inc'],
	library_dirs = ['D:/work_center/sw_dev/python/basic/src/Python-3.2.2/PCBuild', 'D:/work_center/sw_dev/cpp/ext/lib'],
)


setup(
	name = "extending",
	version = "1.0",
	description = "A simple extension module",
    author = "Sang-Wook Lee",
	author_email = "sangwook236@gmail.com",
	url = "http://www.sangwook.com",
    ext_modules = [simple_extending_mod, greeting_mod, greeting_using_boost_mod],
)
