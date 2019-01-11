from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os, subprocess
import numpy as np

# REF [site] >>
#	https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html

def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

def locate_cuda():
    if 'posix' == os.name:
        return locate_cuda_in_linux()
    else:
        return locate_cuda_in_windows()

def locate_cuda_in_linux():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use.
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC.
        default_path = os.path.join(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': os.path.join(home, 'include'),
                  'lib64': os.path.join(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

def locate_cuda_in_windows():
    if 'CUDA_PATH' in os.environ:
        home = os.environ['CUDA_PATH']
        nvcc = os.path.join(home, 'bin', 'nvcc')
        cudaconfig = {'home': home, 'nvcc': nvcc,
                      'include': os.path.join(home, 'include'),
                      'lib64': os.path.join(home, 'lib', 'x64')}
        return cudaconfig
    else:
        raise EnvironmentError('The environment variable CUDA_PATH not found.')

CUDA = locate_cuda()

# Compile nms_kernel.cu.
if 'posix' == os.name:
	subprocess.call([CUDA['nvcc'], '-c', 'nms_kernel.cu', 'parallel_nms_gpu.cu', '-arch=sm_35', '--ptxas-options=-v', '--compiler-options=-fPIC'])
	gpu_nms_kernel_obj = 'nms_kernel.o'
	parallel_nms_gpu_obj = 'parallel_nms_gpu.o'
else:
	subprocess.call([CUDA['nvcc'], '-c', 'nms_kernel.cu', 'parallel_nms_gpu.cu', '--compiler-options=/nologo,/Ox,/W3,/GL,/DNDEBUG,/MD,/EHsc'])
	gpu_nms_kernel_obj = 'nms_kernel.obj'
	parallel_nms_gpu_obj = 'parallel_nms_gpu.obj'

ext_modules=[
	Extension(
		'cpu_nms',
		sources=['cpu_nms.pyx'],
		include_dirs=[numpy_include],
		language='c++',
	),
	Extension(
		'gpu_nms',
		sources=['gpu_nms.pyx'],
		extra_objects=[gpu_nms_kernel_obj],
		include_dirs=[numpy_include, CUDA['include']],
		library_dirs=[CUDA['lib64']],
		libraries=['cudart'],
		language='c++',
	),
	Extension(
		'py_parallel_nms_cpu',
		sources=['py_parallel_nms_cpu.pyx'],
		include_dirs=[numpy_include],
		language='c++',
	),
	Extension(
		'py_parallel_nms_gpu',
		sources=['py_parallel_nms_gpu.pyx'],
		extra_objects=[parallel_nms_gpu_obj],
		include_dirs=[numpy_include, CUDA['include']],
		library_dirs=[CUDA['lib64']],
		libraries=['cudart'],
		language='c++',
	),
]

# Usage:
#	REF [site] >> http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html
#	python setup.py build_ext --inplace

setup(
	name='nms_test',
	ext_modules=cythonize(ext_modules, annotate=False, quiet=True),
)
