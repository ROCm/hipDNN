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
target_compile_options(hipdnn_frontend INTERFACE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_include_directories(hipdnn_frontend INTERFACE
    $<BUILD_INTERFACE:${HIP_DNN_FRONTEND_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

rocm_install_targets(
    TARGETS hipdnn_frontend
    INCLUDE ${HIP_DNN_FRONTEND_INCLUDE_DIR}
)

rocm_create_package(
    NAME hipDNN-frontend
    DESCRIPTION "AMD DNN Library"
    MAINTAINER "hipDNN-frontend Maintainer"
    HEADER_ONLY
)