# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(HIP_DNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

option(HIP_DNN_FRONTEND_BUILD_PYTHON_BINDINGS "Defines if python bindings are built or not." OFF)

if (HIP_DNN_FRONTEND_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
add_subdirectory(tests)

add_library(hipdnn_frontend INTERFACE)
target_link_libraries(hipdnn_frontend INTERFACE hipdnn_sdk)

SET(HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/frontend")

target_include_directories(hipdnn_frontend INTERFACE
    $<BUILD_INTERFACE:${HIP_DNN_FRONTEND_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}/include>
)

install( 
    TARGETS hipdnn_frontend hipdnn_sdk FlatBuffers
    EXPORT hipdnn-frontend
)

install(
    DIRECTORY ${HIP_DNN_FRONTEND_INCLUDE_DIR}
    DESTINATION ${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}
)

install(EXPORT hipdnn-frontend
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/hipdnn_frontend
    FILE hipdnn_frontendConfig.cmake
)