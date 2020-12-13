set(COMPONENT_NAME tpstorch)

#### Add C++ Source Files
set(_${COMPONENT_NAME}_sources
    module-tpstorch.cpp
    )

### Finding External Packages
### Only if we need more. Importing Torch already gives us
### Access to many packages neeed
#Find MPI
find_package(MPI REQUIRED)

pybind11_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} NO_EXTRAS)
target_link_libraries(_${COMPONENT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

#Link our .py scripts so that CMake updates them for newer builds
set(files   __init__.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
       )

foreach(file ${files})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
endforeach()

add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${files})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/extern/)

add_subdirectory(fts)
add_subdirectory(ml)
add_subdirectory(sm)
