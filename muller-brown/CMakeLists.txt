project(mullerbrown)

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_VERBOSE_MAKEFILE on )
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_NO_BUILTIN_CHRPATH TRUE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#Finding TPSTorch
include(FindTPSTorch.cmake)
#Setting Up Torch
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "Found Torch Python Library: ${TORCH_PYTHON_LIBRARY}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Plugins must be built as shared libraries
if (ENABLE_STATIC)
    message(SEND_ERROR "Needs to be built as shared library!")
endif (ENABLE_STATIC)

set(BUILD_SHARED_LIBS on)

#### Add C++ Source Files
set(${CMAKE_PROJECT_NAME}_sources
    muller_brown.cpp
    mysampler.cpp
    )

#add_executable is what you normally use as the compiled object. For Python modules, we use pybind11_add_module instead
#add_executable(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_sources})
pybind11_add_module(${CMAKE_PROJECT_NAME} SHARED ${${CMAKE_PROJECT_NAME}_sources} NO_EXTRAS)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${TPSTORCH_LIBRARIES})
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 14)
