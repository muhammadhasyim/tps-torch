# CMake script for finding TORCH and setting up all needed compile options to create and link a plugin library
#
# Variables taken as input to this module:
# TORCH_ROOT :          location to look for TORCH, if it is not in the python path
#
# Variables defined by this module:
# FOUND_TORCH :         set to true if TORCH is found

set(TORCH_ROOT "" CACHE FILEPATH "Directory containing a torch installation (i.e. _torch.so)")

# Let TORCH_ROOT take precedence, but if unset, try letting Python find a torch package in its default paths.
if(TORCH_ROOT)
  set(torch_installation_guess ${TORCH_ROOT})
else(TORCH_ROOT)
  find_package(PythonInterp 3)

  set(find_torch_script "
from __future__ import print_function;
import sys, os; sys.stdout = open(os.devnull, 'w')
import torch
print(os.path.dirname(torch.__file__), file=sys.stderr, end='')")

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${find_torch_script}"
                  ERROR_VARIABLE torch_installation_guess)
  message(STATUS "Python output: " ${torch_installation_guess})
endif(TORCH_ROOT)

message(STATUS "Looking for a TORCH installation at " ${torch_installation_guess})
find_path(FOUND_TORCH_ROOT
        NAMES __init__.py
        HINTS ${torch_installation_guess}
        )

if(FOUND_TORCH_ROOT)
    set(TORCH_ROOT ${FOUND_TORCH_ROOT} CACHE FILEPATH "Directory containing a torch installation (i.e. __init__.py)" FORCE)
    message(STATUS "Found torch installation at " ${TORCH_ROOT})
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_ROOT})
else(FOUND_TORCH_ROOT)
    message(FATAL_ERROR "Could not find torch installation, either set TORCH_ROOT or set PYTHON_EXECUTABLE to a python which can find torch")
endif(FOUND_TORCH_ROOT)
