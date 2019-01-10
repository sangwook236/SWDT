from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os, subprocess
import numpy as np

# REF [site] >>
#	https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

def locate_cuda():
    if 'posix' == os.name:
        return locate_cuda_in_unix_like_system()
    else:
        return locate_cuda_in_windows()

def locate_cuda_in_unix_like_system():
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
    for k, v in cudaconfig.iteritems():
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
	subprocess.call([CUDA['nvcc'], '-c', 'nms_kernel.cu'])
	nms_kernel_obj = 'nms_kernel.o'
else:
	subprocess.call([CUDA['nvcc'], '-c', 'nms_kernel.cu', '--compiler-options=/nologo,/Ox,/W3,/GL,/DNDEBUG,/MD,/EHsc'])
	nms_kernel_obj = 'nms_kernel.obj'

ext_modules=[
	Extension(
		'cpu_nms',
		sources=['cpu_nms.pyx'],
		include_dirs=[numpy_include]
	),
	Extension(
		'gpu_nms',
		sources=['gpu_nms.pyx'],
		extra_objects=[nms_kernel_obj],
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
