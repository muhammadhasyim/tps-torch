set(SUBCOMPONENT_NAME fts)

#### Add C++ Source Files
set(_${SUBCOMPONENT_NAME}_sources
    module-fts.cpp
    )

pybind11_add_module(_${SUBCOMPONENT_NAME} SHARED ${_${SUBCOMPONENT_NAME}_sources} NO_EXTRAS)
target_link_libraries(_${SUBCOMPONENT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

set(files   __init__.py
    )

foreach(file ${files})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
endforeach()

add_custom_target(copy_${SUBCOMPONENT_NAME} ALL DEPENDS ${files})
