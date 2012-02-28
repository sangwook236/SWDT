@echo off

rem bootstrap.bat

rem bjam {stage/install/clean} release debug --toolset=msvc-8.0 optimization={off/full/space/speed} link=static,shared threading=single,multi runtime-link=single,shared --build-type=complete

rem bjam stage --toolset=msvc link=static,shared --with-thread --with-regex --with-python
rem bjam stage --toolset=msvc link=static,shared --without-python --without-mpi
rem bjam stage -sICU_PATH=%ICU_ROOT% --toolset=msvc link=static,shared

rem bjam stage release debug --toolset=msvc-8.0 link=static,shared --build-type=complete --without-python --without-mpi
bjam stage release debug --toolset=msvc link=static,shared --build-type=complete --without-mpi

rem bjam install --prefix=/bin/local
rem bjam clean release debug

rem %BOOST_ROOT%/tools/build/v2/user-config.jam

rem # -------------------
rem # MSVC configuration.
rem # -------------------

rem # Configure specific msvc version (searched for in standard locations and PATH).
rem # using msvc : 8.0 : "C:/Program Files/Microsoft Visual Studio 8/VC/bin/cl" ;

rem # ---------------------
rem # Python configuration.
rem # ---------------------

rem using python : 3.1 : "C:/Program Files/Python31" : "C:/Program Files/Python31/Include" : "C:/Program Files/Python31/libs" ;

rem using mpi ;

echo on
