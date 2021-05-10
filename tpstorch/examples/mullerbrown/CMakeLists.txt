set(SUBCOMPONENT_NAME mullerbrown)

#### Add C++ Source Files
set(_${SUBCOMPONENT_NAME}_sources
    muller_brown.cpp
    mysampler.cpp
    )

#### Add C++ Header Files
set(_${SUBCOMPONENT_NAME}_headers
    muller_brown.h
    mysampler.h
    )

include_directories(${MPI_CXX_INCLUDE_DIRS})
pybind11_add_module(_${SUBCOMPONENT_NAME} SHARED ${_${SUBCOMPONENT_NAME}_sources} NO_EXTRAS)
target_link_libraries(_${SUBCOMPONENT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${MPI_CXX_LIBRARIES})

set(files   __init__.py
    )

foreach(file ${files})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
endforeach()

add_custom_target(copy_${SUBCOMPONENT_NAME} ALL DEPENDS ${files})

# copy header files 
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/tpstorch/examples/mullerbrown/${file}
        COMMENT    "Copy tpstorch/include/tpstorch/examples/mullerbrown/${file}"
    )
ENDMACRO(copy_header)

# copy headers into build directory
foreach(file ${_${SUBCOMPONENT_NAME}_headers})
	copy_header(${file})
endforeach()

add_custom_target(copy_${SUBCOMPONENT_NAME}_headers ALL DEPENDS ${_${SUBCOMPONENT_NAME}_headers})
