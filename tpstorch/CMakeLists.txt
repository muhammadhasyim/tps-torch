set(COMPONENT_NAME tpstorch)

#### Add C++ Source Files
set(_${COMPONENT_NAME}_sources
    module-tpstorch.cpp
    )

#### Finding External Packages
#Find MPI
#find_package(MPI REQUIRED)
#list(APPEND LIBRARIES ${MPI_CXX_LIBRARIES})

#set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..")
#find_package(Torch REQUIRED)
pybind11_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} NO_EXTRAS)
target_link_libraries(_${COMPONENT_NAME} PRIVATE "${TORCH_LIBRARIES}")
#target_link_libraries(_${COMPONENT_NAME} PRIVATE ${LIBRARIES})
add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${files})

#Link our .py scripts so that CMake updates them for newer builds
set(files   __init__.py
#            utils.py
#            io.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
       )

foreach(file ${files})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
endforeach()
