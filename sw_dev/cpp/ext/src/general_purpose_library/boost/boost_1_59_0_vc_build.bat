@echo off

rem bootstrap.bat

rem b2 {stage/install/clean} release debug --toolset=msvc-14.0 optimization={off/full/space/speed} link=static,shared threading=single,multi runtime-link=single,shared --build-type=complete

rem b2 stage --toolset=msvc link=static,shared --with-thread --with-regex --with-python
rem b2 stage --toolset=msvc link=static,shared --without-python --without-mpi
rem b2 stage -sICU_PATH=%ICU_ROOT% --toolset=msvc link=static,shared

rem b2 stage release debug --toolset=msvc-14.0 link=static,shared --build-type=complete --without-python --without-mpi
b2 stage release debug --toolset=msvc link=static,shared --build-type=complete --without-mpi

rem b2 install --prefix=/bin/local
rem b2 clean release debug

rem %BOOST_ROOT%/tools/build/v2/user-config.jam

rem # -------------------
rem # MSVC configuration.
rem # -------------------

rem # Configure specific msvc version (searched for in standard locations and PATH).
rem # using msvc : 14.0 : "C:/Program Files (x86)/Microsoft Visual Studio 14/VC/bin/cl" ;

rem # ---------------------
rem # Python configuration.
rem # ---------------------

rem using python : 3.5 : "D:/MyProgramFiles/Python35" : "D:/MyProgramFiles/Python35/include" : "D:/MyProgramFiles/Python35/libs" ;

rem using mpi ;

echo on
