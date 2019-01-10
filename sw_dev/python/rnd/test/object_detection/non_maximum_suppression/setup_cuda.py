# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# REF [site] >>
#	https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
#	https://github.com/MrGF/py-faster-rcnn-windows/blob/master/lib/setup.py
#	https://github.com/cudamat/cudamat/blob/6565e63a23a2d61b046b8d115346130da05e7d31/setup.py
#	https://github.com/peterwittek/somoclu/blob/master/src/Python/setup.py

import os, sys
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from distutils.spawn import spawn, find_executable
from Cython.Distutils import build_ext
import subprocess
import numpy as np

def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

#change for windows, by MrX
nvcc_bin = 'nvcc.exe'
cl_bin = 'cl.exe'
PATH = os.environ.get('PATH')

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

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

def locate_cuda_in_windows():
    if 'CUDA_PATH' in os.environ:
        home = os.environ['CUDA_PATH']
        nvcc = pjoin(home, 'bin', 'nvcc')
        cudaconfig = {'home': home, 'nvcc': nvcc,
                      'include': pjoin(home, 'include'),
                      'lib64': pjoin(home, 'lib', 'x64')}
        return cudaconfig
    else:
        raise EnvironmentError('The environment variable CUDA_PATH not found.')

CUDA = locate_cuda()

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
class custom_build_ext_cuda_linux(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# run the customize_compiler
class custom_build_ext_cuda_windows(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        if hasattr(self.compiler, '_c_extensions'):
            self.compiler._cpp_extensions.append('.cu')  # needed for Windows
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)

    def spawn(self, cmd, search_path=1, verbose=0, dry_run=0):
        """
        Perform any CUDA specific customizations before actually launching
        compile/link etc. commands.
        """
        if (sys.platform == 'darwin' and len(cmd) >= 2 and cmd[0] == 'nvcc' and
            cmd[1] == '--shared' and cmd.count('-arch') > 0):
            # Versions of distutils on OSX earlier than 2.7.9 inject
            # '-arch x86_64' which we need to strip while using nvcc for
            # linking
            while True:
                try:
                    index = cmd.index('-arch')
                    del cmd[index:index+2]
                except ValueError:
                    break
        elif self.compiler.compiler_type == 'msvc':
            is_cu = False
            for idx, c in enumerate(cmd):
                if (c.startswith('/Tc') or c.startswith('/Tp')) and c.endswith('.cu'):
                    is_cu = True
                    break

            if is_cu:
                cmd = self.convert_nvcc_options(cmd)
                pass_on = '--compiler-options='
                cmd = ([c for c in cmd if c[0] != '/'] +
                    [pass_on + ','.join(c for c in cmd if c[0] == '/')])
            else:
                cmd[:1] = [cl_bin]

            if '/DLL' in cmd:
                cmd = self.convert_nvcc_options(cmd)
                # we only need MSVCRT for a .dll, remove CMT if it sneaks in:
                cmd.append('/NODEFAULTLIB:libcmt.lib')
                pass_on = '--linker-options='
                cmd = ([c for c in cmd if c[0] != '/'] +
                    [pass_on + ','.join(c for c in cmd if c[0] == '/')])
        spawn(cmd, search_path, verbose, dry_run)

    def convert_nvcc_options(self, cmd):
        # There are several things we need to do to change the commands
        # issued by MSVCCompiler into one that works with nvcc. In the end,
        # it might have been easier to write our own CCompiler class for
        # nvcc, as we're only interested in creating a shared library to
        # load with ctypes, not in creating an importable Python extension.
        # - First, we replace the cl.exe or link.exe call with an nvcc
        #   call. In case we're running Anaconda, we search cl.exe in the
        #   original search path we captured further above -- Anaconda
        #   inserts a MSVC version into PATH that is too old for nvcc.
        cmd[:1] = [nvcc_bin, '--compiler-bindir',
                   os.path.dirname(find_executable('cl.exe', PATH))
                   or cmd[0]]
        # - Secondly, we fix a bunch of command line arguments.
        for idx, c in enumerate(cmd):
            # create .dll instead of .pyd files
            if '.pyd' in c: cmd[idx] = c = c.replace('.pyd', '.dll')
            # replace /c by -c
            if c == '/c': cmd[idx] = '-c'
            # replace /DLL by --shared
            elif c == '/DLL': cmd[idx] = '--shared'
            # remove --compiler-options=-fPIC
            elif '-fPIC' in c: del cmd[idx]
            # replace /Tc... by ...
            elif c.startswith('/Tc'): cmd[idx] = c[3:]
            # replace /Tp... by ...
            elif c.startswith('/Tp'): cmd[idx] = c[3:]
            # replace /Fo... by -o ...
            elif c.startswith('/Fo'): cmd[idx:idx+1] = ['-o', c[3:]]
            # replace /LIBPATH:... by -L...
            elif c.startswith('/LIBPATH:'): cmd[idx] = '-L' + c[9:]
            # replace /OUT:... by -o ...
            elif c.startswith('/OUT:'): cmd[idx:idx+1] = ['-o', c[5:]]
            # remove /EXPORT:initlibcudamat or /EXPORT:initlibcudalearn
            elif c.startswith('/EXPORT:'): del cmd[idx]
            # replace cublas.lib by -lcublas
            elif c == 'cublas.lib': cmd[idx] = '-lcublas'
            # remove MANIFEST:EMBED
            elif 'MANIFEST:EMBED' in c: del cmd[idx]
        # For the future: Apart from the wrongly set PATH by Anaconda, it
        # would suffice to run the following for compilation on Windows:
        # nvcc -c -O -o <file>.obj <file>.cu
        # And the following for linking:
        # nvcc --shared -o <file>.dll <file1>.obj <file2>.obj -lcublas
        # This could be done by a NVCCCompiler class for all platforms.

        return cmd

ext_modules = [
    Extension(
        'cpu_nms',
        sources=['cpu_nms.pyx'],
        extra_compile_args={cl_bin: []},
        include_dirs=[numpy_include]
    ),
    Extension(
	    'gpu_nms',
        sources=['nms_kernel.cu', 'gpu_nms.pyx'],
        include_dirs=[numpy_include, CUDA['include']]
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        #runtime_library_dirs=[pjoin(CUDA['home'], 'bin')],
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with
        # gcc the implementation of this trick is in customize_compiler() below
        #extra_compile_args={cl_bin: [],
        #                    nvcc_bin: ['-arch=sm_35',
        #                             '--ptxas-options=-v',
        #                             '-c',
        #                             '--compiler-options=-fPIC']},
    ),
]

if 'posix' == os.name:
    cmdclass={'build_ext': custom_build_ext_cuda_linux}
else:
    cmdclass={'build_ext': custom_build_ext_cuda_windows}

setup(
    name='nms_cuda',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass=cmdclass,
)
