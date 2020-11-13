# CMake script for finding TPSTORCH and setting up all needed compile options to create and link a plugin library
#
# Variables taken as input to this module:
# TPSTORCH_ROOT :          location to look for TPSTORCH, if it is not in the python path
#
# Variables defined by this module:
# FOUND_TPSTORCH :         set to true if TPSTORCH is found
# TPSTORCH_INCLUDE_DIR :   a list of all include directories that need to be set to include TPSTORCH

set(TPSTORCH_ROOT "" CACHE FILEPATH "Directory containing a tpstorch installation (i.e. _tpstorch.so)")

# Let TPSTORCH_ROOT take precedence, but if unset, try letting Python find a tpstorch package in its default paths.
if(TPSTORCH_ROOT)
  set(tpstorch_installation_guess ${TPSTORCH_ROOT})
else(TPSTORCH_ROOT)
  find_package(PythonInterp 3)

  set(find_tpstorch_script "
from __future__ import print_function;
import sys, os; sys.stdout = open(os.devnull, 'w')
import tpstorch
print(os.path.dirname(tpstorch.__file__), file=sys.stderr, end='')")

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${find_tpstorch_script}"
                  ERROR_VARIABLE tpstorch_installation_guess)
  message(STATUS "Python output: " ${tpstorch_installation_guess})
endif(TPSTORCH_ROOT)

message(STATUS "Looking for a TPSTORCH installation at " ${tpstorch_installation_guess})
find_path(FOUND_TPSTORCH_ROOT
        NAMES _tpstorch.*so __init__.py
        HINTS ${tpstorch_installation_guess}
        )

if(FOUND_TPSTORCH_ROOT)
  set(TPSTORCH_ROOT ${FOUND_TPSTORCH_ROOT} CACHE FILEPATH "Directory containing a tpstorch installation (i.e. _tpstorch.so)" FORCE)
  message(STATUS "Found tpstorch installation at " ${TPSTORCH_ROOT})
else(FOUND_TPSTORCH_ROOT)
  message(FATAL_ERROR "Could not find tpstorch installation, either set TPSTORCH_ROOT or set PYTHON_EXECUTABLE to a python which can find tpstorch")
endif(FOUND_TPSTORCH_ROOT)

# search for the tpstorch include directory
set(TPSTORCH_INCLUDE_DIR ${TPSTORCH_ROOT}/include)
if (TPSTORCH_INCLUDE_DIR)
    message(STATUS "Found TPSTORCH include directory: ${TPSTORCH_INCLUDE_DIR}")
    mark_as_advanced(TPSTORCH_INCLUDE_DIR)
endif (TPSTORCH_INCLUDE_DIR)

set(TPSTORCH_FOUND FALSE)
if (TPSTORCH_INCLUDE_DIR AND TPSTORCH_ROOT)
    set(TPSTORCH_FOUND TRUE)
    mark_as_advanced(TPSTORCH_ROOT)
endif (TPSTORCH_INCLUDE_DIR AND TPSTORCH_ROOT)

if (NOT TPSTORCH_FOUND)
    message(SEND_ERROR "TPSTORCH Not found. Please specify the location of your tpstorch installation in TPSTORCH_ROOT")
endif (NOT TPSTORCH_FOUND)

#############################################################
## Now that we've found tpstorch, lets do some setup
list(APPEND CMAKE_MODULE_PATH 	${TPSTORCH_ROOT})
list(APPEND CMAKE_MODULE_PATH 	${TPSTORCH_ROOT}/CMake)
#Include directories for TPSTorch
include_directories(${TPSTORCH_INCLUDE_DIR})
#Set up variables for TPSTorch Libraries
set(TPSTORCH_LIB ${TPSTORCH_ROOT}/_tpstorch${PYTHON_MODULE_EXTENSION})
set(TPSTORCH_FTS_LIB ${TPSTORCH_ROOT}/fts/_fts${PYTHON_MODULE_EXTENSION})
set(TPSTORCH_LIBRARIES ${TPSTORCH_LIB} ${TPSTORCH_FTS_LIB})

#Now, Find pybind11, which should be installed along with TPSTorch
set(find_pybind11_script "
from __future__ import print_function;
import sys, os; sys.stdout = open(os.devnull, 'w')
import pybind11
print(os.path.dirname(pybind11.__file__), file=sys.stderr, end='')")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${find_pybind11_script}"
                ERROR_VARIABLE pybind11_dir)

set(PYBIND11_CMAKE_ROOT ${pybind11_dir}/share/cmake/pybind11)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_ROOT})
find_package(pybind11 CONFIG REQUIRED)

#Finally, look for Torch
#list(APPEND CMAKE_PREFIX_PATH ${TPSTORCH_ROOT}/extern/pybind11/include)
include(GetTorchPath)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "Found Torch Python Library: ${TORCH_PYTHON_LIBRARY}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

