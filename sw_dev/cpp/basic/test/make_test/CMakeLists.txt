# Usage:
#	Configure and generate:
#		mkdir build
#		cd build
#		cmake --help
#		cmake .. -G Ninja
#		cmake .. -G "Visual Studio 17 2022" -T v143
#		cmake .. -G "Unix Makefiles"
#		cmake .. -D CMAKE_PREFIX_PATH=/path/to/lib -D BOOST_LIBRARYDIR:PATH=/usr/lib
#	Build:
#		cmake --build . --parallel
#		cmake --build . --config Release --target all -j 14 --
#		cmake --build . --config Release --target ALL_BUILD
#		cmake --build . --target INSTALL -- /verbosity:detailed
#			Pass additional options to the native build tool with the -- switch.
#	Run:
#		./my_cpp

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(my_cpp VERSION 0.1.0)

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
endif()
#set(CMAKE_DEBUG_POSTFIX d)

if(APPLE)
elseif(CYGWIN)
elseif(MSYS OR MINGW)
elseif(UNIX AND NOT APPLE AND NOT CYGWIN)
	set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
#	set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
elseif(WIN32 AND NOT CYGWIN)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#	set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -Wno-deprecated-declarations")
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
#	set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
#	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

	add_definitions(-DUNICODE -D_UNICODE)

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

	add_definitions(/DUNICODE /D_UNICODE)
endif()

#--------------------
find_package(Threads REQUIRED)
#find_package(OpenMP REQUIRED)
#find_package(OpenGL REQUIRED)
#find_package(Boost 1.86 REQUIRED thread timer)
#find_package(Log4cxx REQUIRED)
#find_package(glog REQUIRED)
#find_package(gflags REQUIRED)
#find_package(GTest REQUIRED)  # ${GTest_FOUND}, ${GMock_FOUND}
#find_package(ZLIB REQUIRED)
#find_package(CUDA REQUIRED)
#find_package(GSL REQUIRED)
#find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)  # ${Qt5_FOUND}, ${Qt5Core_FOUND}, ${Qt5Gui_FOUND}, ${Qt5Widgets_FOUND}
#find_package(OpenCV 4.10.0 REQUIRED)
#message(STATUS "Boost found: ${Boost_FOUND}")

#find_library(MY_LIBRARIES NAMES libmy.a)
#find_library(MY_LIBRARIES NAMES libmy.a PATHS /path/to/lib)
#message(STATUS "MY_LIBRARIES found: ${MY_LIBRARIES}")

#include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${LOG4CXX_INCLUDE_DIR})
#include_directories(${GLOG_INCLUDE_DIR})
#include_directories(${GFLAGS_INCLUDE_DIR})
#include_directories(${GTEST_INCLUDE_DIRS})
#include_directories(${ZLIB_INCLUDE_DIRS})
#include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(/usr/include/eigen3)
#include_directories(${GSL_INCLUDE_DIRS})
#include_directories(${Qt5Core_INCLUDE_DIRS})  # ${Qt5Core_INCLUDE_DIRS}, ${Qt5Gui_INCLUDE_DIRS}, ${Qt5Widgets_INCLUDE_DIRS}
#include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(AFTER /usr/local/include)
#message(STATUS "Include directories: ${INCLUDE_DIRECTORIES}")

#link_directories(${Boost_LIBRARY_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#link_directories(AFTER /usr/local/lib)
#message(STATUS "Link directories: ${LINK_DIRECTORIES}")

#add_definitions(${OpenMP_CXX_FLAGS})
#add_definitions(${Qt5Core_DEFINITIONS})  # ${Qt5Core_DEFINITIONS}, ${Qt5Gui_DEFINITIONS}, ${Qt5Widgets_DEFINITIONS}
#add_definitions(${PCL_DEFINITIONS})
#message(STATUS "Compile definitions: ${COMPILE_DEFINITIONS}")

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	if(OpenMP_CXX_FOUND)
		add_compile_options(-fopenmp)
	endif(OpenMP_CXX_FOUND)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
	if(OpenMP_CXX_FOUND)
		add_compile_options(/fopenmp)
	endif(OpenMP_CXX_FOUND)
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
#	${Boost_LIBRARIES}
#	${LOG4CXX_LIBRARIES}
#	${GLOG_LIBRARIES}
#	${GFLAGS_LIBRARIES}
#	${GTEST_BOTH_LIBRARIES}  # ${GTEST_LIBRARIES}, ${GTEST_MAIN_LIBRARIES}, ${GTEST_BOTH_LIBRARIES}, ${GMOCK_LIBRARIES}, ${GMOCK_MAIN_LIBRARIES}
#	${ZLIB_LIBRARIES}
#	${CUDA_LIBRARIES}
#	Qt5::Core Qt5::Gui Qt5::Widgets
#	${OpenCV_LIBRARIES}
)

add_executable(${TARGET} ${SRCS} ${HDRS})
#add_library(${TARGET} SHARED ${SRCS} ${HDRS})
#add_library(${TARGET} STATIC ${SRCS} ${HDRS})
#cuda_add_executable(${TARGET} ${SRCS} ${HDRS})
#cuda_add_library(${TARGET} SHARED ${SRCS} ${HDRS})
#cuda_add_library(${TARGET} STATIC ${SRCS} ${HDRS})

target_link_libraries(${TARGET} ${LIBS})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

#--------------------
# Testing:
#	ctest --help
#	ctest
#	make test

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

#install(
#	TARGETS ${TARGET}
#	RUNTIME DESTINATION bin
#	LIBRARY DESTINATION lib
#	ARCHIVE DESTINATION lib
#)

# Packaging:
#	cpack --help
#	cpack
#	cpack --config CPackConfig.cmake
#	cpack --config CPackSourceConfig.cmake
#	make package
#	make package_source

#include(CPack)
