# Usage:
#	Configure and generate:
#		mkdir build
#		cd build
#		cmake --help
#		cmake -B /path/to/build -S /path/to/source
#		cmake ..
#		cmake .. -D CMAKE_BUILD_TYPE:STRING=Debug -D CMAKE_PREFIX_PATH:PATH=/path/to/install -D BOOST_LIBRARYDIR=/path/to/boost -D CMAKE_C_COMPILER:FILEPATH=/path/to/clang -D CMAKE_CXX_COMPILER=/path/to/clang++
#		cmake .. -G "Visual Studio 17 2022"
#		cmake .. -G "Visual Studio 17 2022" -A x64 -T v143
#		cmake .. -G "MSYS Makefiles"
#		cmake .. -G "MinGW Makefiles"
#		cmake .. -G "Unix Makefiles"
#		cmake .. -G Ninja
#		cmake .. -G Ninja -D CMAKE_MAKE_PROGRAM=/path/to/ninja
#	Build:
#		cmake --build . --parallel
#		cmake --build . --config Release --target all -j 14 --
#		cmake --build . --config Release --target ALL_BUILD
#		cmake --build . --target INSTALL -- /verbosity:detailed
#			Pass additional options to the native build tool with the -- switch
#		make all
#		ninja all
#	Run:
#		./my_cpp

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(my_cpp VERSION 0.1.0)
#project(
#	my_cpp
#	VERSION 0.1.0
#	DESCRIPTION "My C++ Project"
#	HOMEPAGE_URL "http://www.sangwook.com/"
#	LANGUAGES C CXX CUDA
#)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
#set(CMAKE_PREFIX_PATH "/path/to/lib")

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()
#set(CMAKE_RELEASE_POSTFIX "")
#set(CMAKE_DEBUG_POSTFIX d)

if(APPLE)
elseif(CYGWIN)
elseif(MSYS OR MINGW)
elseif(UNIX AND NOT APPLE AND NOT CYGWIN)
#	set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
#	set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
elseif(WIN32 AND NOT CYGWIN)
#	set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_PATH}")
endif()

#--------------------
find_package(Threads REQUIRED)
#find_package(OpenMP REQUIRED)
#find_package(OpenGL REQUIRED)
#find_package(CUDA REQUIRED)
#find_package(Boost 1.86 REQUIRED system thread timer)
#find_package(Log4cxx REQUIRED)
#find_package(glog REQUIRED)
#find_package(gflags REQUIRED)
#find_package(GTest REQUIRED)  # ${GTest_FOUND}, ${GMock_FOUND}
#find_package(ZLIB REQUIRED)
find_package(Eigen3 REQUIRED)  # ${Eigen3_FOUND}, ${EIGEN3_FOUND}
#find_package(GSL REQUIRED)
#find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)  # ${Qt5_FOUND}, ${Qt5Core_FOUND}, ${Qt5Gui_FOUND}, ${Qt5Widgets_FOUND}
#find_package(OpenCV 4.10.0 REQUIRED)
#message(STATUS "Boost found: ${Boost_FOUND}")

#find_library(MY_LIBRARIES NAMES libmy.a)
#find_library(MY_LIBRARIES NAMES libmy.a PATHS /path/to/lib)
#message(STATUS "MY_LIBRARIES found: ${MY_LIBRARIES}")

#include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${LOG4CXX_INCLUDE_DIR})
#include_directories(${GLOG_INCLUDE_DIR})
#include_directories(${GFLAGS_INCLUDE_DIR})
#include_directories(${GTEST_INCLUDE_DIRS})
#include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})  # ${EIGEN3_INCLUDE_DIR}, ${EIGEN3_INCLUDE_DIRS}
#include_directories(${GSL_INCLUDE_DIRS})
#include_directories(${Qt5Core_INCLUDE_DIRS})  # ${Qt5Core_INCLUDE_DIRS}, ${Qt5Gui_INCLUDE_DIRS}, ${Qt5Widgets_INCLUDE_DIRS}
#include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(AFTER /usr/local/include)
#include_directories(AFTER /usr/include)
#message(STATUS "Include directories: ${INCLUDE_DIRECTORIES}")

#link_directories(${Boost_LIBRARY_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#link_directories(AFTER /usr/local/lib)
#link_directories(AFTER /usr/lib)
#message(STATUS "Link directories: ${LINK_DIRECTORIES}")

add_definitions(-DUNICODE -D_UNICODE)
#add_definitions(${OpenMP_CXX_FLAGS})
#add_definitions(${Qt5Core_DEFINITIONS})  # ${Qt5Core_DEFINITIONS}, ${Qt5Gui_DEFINITIONS}, ${Qt5Widgets_DEFINITIONS}
#add_definitions(${PCL_DEFINITIONS})
#message(STATUS "Compile definitions: ${COMPILE_DEFINITIONS}")

#--------------------
if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
	set(CMAKE_CXX_FLAGS "-target x86_64-w64-mingw32 -std=c++17 -Wall -Wextra -Wpedantic")
#	set(CMAKE_CXX_FLAGS "-target x86_64-w64-mingw32 -std=c++17 -Wall -Wextra -Wpedantic -pthread")
#	set(CMAKE_CXX_FLAGS "-target x86_64-pc-windows-gnu -std=c++17 -Wall -Wextra -Wpedantic -pthread")
#	set(CMAKE_CXX_FLAGS "-target x86_64-pc-windows-msvc -std=c++17 -Wall -Wextra -Wpedantic -pthread")
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
#	set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
#	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

	if(OpenMP_CXX_FOUND)
		add_compile_options(-fopenmp)
	endif(OpenMP_CXX_FOUND)
elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
#	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -pthread")
#	set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -Wpedantic -Wno-deprecated-declarations")
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
#	set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
#	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

	if(OpenMP_CXX_FOUND)
		add_compile_options(-fopenmp)
	endif(OpenMP_CXX_FOUND)

#	add_compile_options(-mavx2)
#	add_compile_options(-fPIC)
#	add_compile_options(-fpermissive)
#	add_compile_options(-Wno-strict-aliasing)
#	add_compile_options(-Wno-unused-function)
#	add_compile_options(-Wno-unused-variable)
#	add_compile_options(-Wno-unused-parameter)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
	set(CMAKE_CXX_FLAGS "/Wall")
#	set(CMAKE_CXX_FLAGS "/std:c++17 /Wall")
	set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1 /D_DEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /D NDEBUG")
#	set(CMAKE_CXX_FLAGS_MINSIZEREL "/MD /O1 /Ob1 /D NDEBUG")
#	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob1 /D NDEBUG")

#	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")  # {-MD, -MDd}. Default.
#	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")  # {-MT, -MTd}.

	if(OpenMP_CXX_FOUND)
		add_compile_options(/fopenmp)
	endif(OpenMP_CXX_FOUND)

#	add_compile_options(/arch:AVX2)
#	add_compile_options(/permissive-)
endif()

#--------------------
#set(TARGET my_cpp)
set(TARGET ${CMAKE_PROJECT_NAME})  # The name of the top level project
#set(TARGET ${PROJECT_NAME})  # The name of the project given to the project command

set(HDRS
#	main.h
)
set(SRCS
	main.cpp
	add.cpp
	sub.cpp
)
set(LIBS
#	${OpenMP_CXX_LIBRARIES}
#	OpenMP::OpenMP_CXX
#	OpenGL::GLU
#	${CUDA_LIBRARIES}
#	${Boost_LIBRARIES}
#	${LOG4CXX_LIBRARIES}
#	${GLOG_LIBRARIES}
#	${GFLAGS_LIBRARIES}
#	${GTEST_BOTH_LIBRARIES}  # ${GTEST_LIBRARIES}, ${GTEST_MAIN_LIBRARIES}, ${GTEST_BOTH_LIBRARIES}, ${GMOCK_LIBRARIES}, ${GMOCK_MAIN_LIBRARIES}
#	${ZLIB_LIBRARIES}
#	Eigen3::Eigen
#	Qt5::Core Qt5::Gui Qt5::Widgets
#	${OpenCV_LIBRARIES}
)

add_executable(${TARGET} ${SRCS} ${HDRS})
#add_library(${TARGET} SHARED ${SRCS} ${HDRS})
#add_library(${TARGET} STATIC ${SRCS} ${HDRS})
#cuda_add_executable(${TARGET} ${SRCS} ${HDRS})
#cuda_add_library(${TARGET} SHARED ${SRCS} ${HDRS})
#cuda_add_library(${TARGET} STATIC ${SRCS} ${HDRS})

#target_link_libraries(${TARGET} ${LIBS})
target_link_libraries(${TARGET} PRIVATE ${LIBS})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

#--------------------
# Testing:
#	ctest --help
#	ctest
#	make test
#	ninja test

#enable_testing()
#include(GoogleTest)

#add_test(NAME my_test COMMAND ${TARGET})
#gtest_add_tests(
#	TARGET ${TARGET}
#	TEST_SUFFIX .noArgs
#	TEST_LIST noArgsTests
#)
#gtest_add_tests(
#	TARGET ${TARGET}
#	EXTRA_ARGS --someArg someValue
#	TEST_SUFFIX .withArgs
#	TEST_LIST withArgsTests
#)
#gtest_discover_tests(
#	${TARGET}
#	TEST_SUFFIX .noArgs
#	TEST_LIST noArgsTests
#)
#gtest_discover_tests(
#	${TARGET}
#	EXTRA_ARGS --someArg someValue
#	TEST_SUFFIX .withArgs
#	TEST_LIST withArgsTests
#	DISCOVERY_TIMEOUT 10
#	DISCOVERY_EXTRA_ARGS --someArg someValue
#)

#set_tests_properties(${noArgsTests} PROPERTIES TIMEOUT 10)
#set_tests_properties(${withArgsTests} PROPERTIES TIMEOUT 20)

#--------------------
# Installation:
#	make install
#	ninja install

#install(
#	TARGETS ${TARGET}
#	RUNTIME DESTINATION bin
#	LIBRARY DESTINATION lib
#	ARCHIVE DESTINATION lib
#)
#file(GLOB_RECURSE HEADER_FILES "${CMAKE_SOURCE_DIR}/include/*.h")
#install(FILES ${HEADER_FILES} DESTINATION include)
#file(GLOB EXECUTABLE_FILES "${CMAKE_SOURCE_DIR}/bin/*")
#install(PROGRAMS ${EXECUTABLE_FILES} DESTINATION bin)

# Packaging:
#	cpack --help
#	cpack
#	cpack --config CPackConfig.cmake
#	cpack --config CPackSourceConfig.cmake
#	make package
#	make package_source
#	ninja package
#	ninja package_source

#include(CPack)
