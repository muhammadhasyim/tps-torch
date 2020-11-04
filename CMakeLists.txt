project(tpstorch)

cmake_minimum_required(VERSION 2.8.12)

set( CMAKE_VERBOSE_MAKEFILE on )
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_NO_BUILTIN_CHRPATH TRUE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tpstorch/extern/pybind11/include)
#include(SetupPybind11.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tpstorch/extern/pybind11)
include(CompilerFlagsSetup.cmake)
include(GetTorchPath.cmake)
find_package(Torch REQUIRED)

# plugins must be built as shared libraries
if (ENABLE_STATIC)
    message(SEND_ERROR "Needs to be built as shared library!")
endif (ENABLE_STATIC)

set(BUILD_SHARED_LIBS on)

# include current source directory as its own include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add subdirectories to process CMake files inside
add_subdirectory(${PROJECT_NAME})

#copy setup.py do I need it?
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py COPYONLY)
