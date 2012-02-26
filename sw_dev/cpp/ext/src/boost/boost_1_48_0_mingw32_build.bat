@echo off

rem ---------------------------------------
rem set path=${MINGW32_ROOT}/bin;%path%

rem bootstrap.bat

rem bjam {stage/install/clean} debug release --toolset=gcc optimization={off/full/space/speed} link=static,shared threading=single,multi runtime-link=single,shared --build-type=complete

rem bjam stage --toolset=gcc link=static,shared --with-thread --with-regex --with-python
rem bjam stage --toolset=gcc link=static,shared --without-python --without-mpi
rem bjam stage -sICU_PATH=${ICU_ROOT} --toolset=gcc link=static,shared

rem bjam stage debug release --toolset=gcc link=static,shared --build-type=complete --without-python --without-mpi
bjam stage debug release --toolset=gcc link=static,shared --build-type=complete --without-python --without-mpi

rem bjam install --prefix=/bin/local --toolset=gcc --without-python --without-mpi
rem bjam --clean debug release

rem %BOOST_ROOT%/tools/build/v2/user-config.jam

rem # -------------------
rem # GCC configuration.
rem # -------------------

rem # Configure specific gcc version (searched for in standard locations and PATH).
rem # using gcc : 4.6.1 ;
rem # using gcc : 4.6.1 : [c++-compile-command] : [compiler options] ;

rem # ---------------------
rem # Python configuration.
rem # ---------------------

rem using python : 3.1 : "C:/Program Files/Python31" : "C:/Program Files/Python31/Include" : "C:/Program Files/Python31/libs" ;

rem using mpi ;

echo on
