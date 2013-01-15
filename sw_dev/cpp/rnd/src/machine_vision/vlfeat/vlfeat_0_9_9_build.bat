rem mkdir bin/w32
rem mkdir bin/w64
rem mkdir toolbox/mex/mexw32
rem mkdir toolbox/mex/mexw64

nmake -f Makefile_release.mak all
nmake -f Makefile_debug.mak all
rem nmake -f Makefile_debug.mak all ARCH=w64
rem nmake -f Makefile_debug.mak all ARCH=w64
