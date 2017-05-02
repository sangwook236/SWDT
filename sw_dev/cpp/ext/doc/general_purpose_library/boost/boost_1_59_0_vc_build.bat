@echo off
setlocal

rem step 1 ----------------------------------------------------------
rem bootstrap.bat

rem step 2 ----------------------------------------------------------
rem add the content below to ${BOOST_ROOT}/project-config.jam
rem REF [site] >>
rem		http://www.boost.org/build/doc/html/index.html
rem		http://www.boost.org/build/doc/html/bbv2/reference/tools.html

rem using msvc : 14.0 : "C:/Program Files (x86)/Microsoft Visual Studio 14/VC/bin/cl" ;
rem using zlib : 1.2.8 : "D:/usr/local/include" "D:/usr/local/lib" ;
rem using python : 3.5 : "D:/MyProgramFiles/Python35" : "D:/MyProgramFiles/Python35/include" : "D:/MyProgramFiles/Python35/libs" ;
rem using mpi ;

rem step 3 ----------------------------------------------------------
rem INFO [option] >>
rem b2 stage|install|clean release debug toolset=msvc-14.0 address-model=32|64 variant=debug|release link=static|shared threading=single|multi runtime-link=single|shared optimization=off|full|space|speed --build-type=complete
rem INFO [example] >>
rem b2 stage toolset=msvc link=static,shared --without-thread --without-regex --without-python
rem b2 stage --stagedir=stage64 toolset=msvc address-model=64 link=static,shared --without-python --without-mpi
rem b2 stage -sICU_PATH=${ICU_ROOT} -sZLIB_SOURCE=${ZLIB_ROOT} toolset=msvc link=static,shared

b2 stage -j4 toolset=msvc variant=release,debug link=static,shared threading=multi --build-type=complete --without-mpi -sICU_PATH="D:/lib_repo/cpp/ext/icu4c-57_1-src/icu" -sICU_LINK="-LD:/usr/local/lib" -sZLIB_SOURCE="D:/lib_repo/cpp/ext/zlib-1.2.8"
rem b2 stage -j4 --stagedir=stage64 toolset=msvc address-model=64 variant=release,debug link=static,shared threading=multi --build-type=complete --without-mpi -sICU_PATH="D:/lib_repo/cpp/ext/icu4c-57_1-src/icu" -sICU_LINK="-LD:/usr/local64/lib" -sZLIB_SOURCE="D:/lib_repo/cpp/ext/zlib-1.2.8"

rem step 4 ----------------------------------------------------------
rem b2 install --prefix=/usr/local

rem step 5 ----------------------------------------------------------
rem b2 clean release debug

endlocal
echo on
