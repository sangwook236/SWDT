rem mkdir bin/w32
rem mkdir bin/w64
rem mkdir toolbox/mex/mexw32
rem mkdir toolbox/mex/mexw64

nmake -f Makefile_msvc10.mak all
rem nmake -f Makefile_msvc10.mak all ARCH=w64
