# This is an example of a distutils 'setup' script for the example_nt
# sample.  This provides a simpler way of building your extension
# and means you can avoid keeping MSVC solution files etc in source-control.
# It also means it should magically build with all compilers supported by
# python.

# USAGE: you probably want 'setup.py install' - but execute 'setup.py --help'
# for all the details.

# NOTE: This is *not* a sample for distutils - it is just the smallest
# script that can build this.  See distutils docs for more info.

# REF [site] >>
#	https://docs.python.org/3/extending/building.html
#	http://en.wikibooks.org/wiki/Python_Programming/Extending_with_C%2B%2B
#	https://docs.python.org/3/distutils/setupscript.html
#	https://docs.python.org/3/distutils/apiref.html
#	https://www.programcreek.com/python/example/88146/distutils.extension.Extension
#	https://www.programcreek.com/python/example/100031/setuptools.command.build_ext.build_ext.build_extensions
#	https://github.com/pytorch/pytorch/blob/master/setup.py

from distutils.core import setup, Extension
import os, sysconfig

_DEBUG = False
_DEBUG_LEVEL = 0

# Common flags for both release and debug builds.
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ['-std=c++11', '-Wall', '-Wextra']
if _DEBUG:
    extra_compile_args += ['-g3', '-O0', '-DDEBUG=%s' % _DEBUG_LEVEL, '-UNDEBUG']
else:
    extra_compile_args += ['-DNDEBUG', '-O3']
extra_link_args = sysconfig.get_config_var('LDFLAGS').split()

"""
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.develop
import setuptools.command.build_py
import distutils.command.build
import distutils.command.clean

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

class custom_build_ext(build_ext):
    def build_extensions(self):
		self.compiler.src_extensions.append('.cu')
		self.compiler.set_executable('compiler_so', 'nvcc')
		self.compiler.set_executable('linker_so', 'nvcc --shared')
		if hasattr(self.compiler, '_c_extensions'):
			self.compiler._c_extensions.append('.cu')  # Needed for Windows.
		self.compiler.spawn = self.spawn
		build_ext.build_extensions(self) 

	def spawn(self, cmd, search_path=1, verbose=0, dry_run=0):
		# Do something.
		spawn(cmd, search_path, verbose, dry_run)

cmdclass = {
    'build': build,
    'build_py': build_py,
    #'build_ext': build_ext,
    'build_ext': custom_build_ext,
    'rebuild': rebuild,
    'develop': develop,
    'install': install,
    'clean': clean,
}
"""

ext_modules = [
	Extension(
		'simple_extending',
		sources=['simple_extending.cpp'],
		include_dirs=['D:/util/Anaconda3/include'],
		library_dirs=['D:/util/Anaconda3/libs'],
		#libraries=['python36'],
	)
	Extension(
		'greeting',
		sources=['greeting.cpp'],
		include_dirs=['D:/util/Anaconda3/include'],
		library_dirs=['D:/util/Anaconda3/libs'],
		#libraries=['python36'],
	)
	Extension(
		'greeting_using_boost',
		sources=['greeting_using_boost.cpp'],
		#extra_objects=['foo.obj'],
		#define_macros=[('PY_MAJOR_VERSION', '3'), ('PY_MINOR_VERSION', '6')],
		#include_dirs=['D:/util/Anaconda3/include', 'D:/usr/local/include', numpy_include],
		include_dirs=['D:/util/Anaconda3/include', 'D:/usr/local/include'],
		library_dirs=['D:/util/Anaconda3/libs', 'D:/usr/local/lib'],
		#libraries=['python36', 'boost_python36-vc141-mt-x64-1_67'],
		extra_compile_args=extra_compile_args,
		extra_link_args=extra_link_args,
		language='c++11',
	),
]

# Usage:
#	python setup.py build
#		.so (Linux) or .pyd (Windows) files are generated.
#		APIs can be used in Python if there are each .pyx file and its corresponding binary file (.so or .pyd file).
#		Do not need to build any additional shared or static library.
#
#	python setup.py install
#		Its installation directory is created in a sub-directory Lib/site-packages of the executed Python interpreter.
#	python setup.py --help

setup(
	name='extending',
	version='1.0',
	description='A simple extension module',
    author='Sang-Wook Lee',
	author_email='sangwook236@gmail.com',
	url='http://www.sangwook.com/',
    ext_modules=ext_modules,
    #cmdclass=cmdclass,
)
