@echo off

rem bootstrap.bat

rem bjam --toolset=msvc-8.0 debug release optimization={off/full/space/speed} link=static,shared threading=single,multi runtime-link=single,shared --build-type=complete {stage/install/clean}
rem bjam --toolset=msvc link=static,shared --with-thread --with-regex --with-python stage
rem bjam --toolset=msvc link=static,shared --without-python --without-mpi stage
rem bjam -sICU_PATH=%ICU_ROOT% --toolset=msvc link=static,shared stage

rem bjam --toolset=msvc-8.0 debug release link=static,shared --build-type=complete --without-python --without-mpi stage
bjam --toolset=msvc debug release link=static,shared --build-type=complete --without-mpi stage

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
