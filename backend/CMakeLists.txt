# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(HIP_DNN_BACKEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/backend/include")

add_subdirectory(src)
add_subdirectory(tests)


install(
    TARGETS hipdnn_backend
    EXPORT hipdnn_backendTargets
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include/hipdnn_backend
)

install(
    DIRECTORY ${HIP_DNN_BACKEND_INCLUDE_DIR}/
    DESTINATION include/hipdnn_backend
    FILES_MATCHING PATTERN "*.h"
)

install(
    FILES
        ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}/hipdnn_backend_export.h
    DESTINATION include/hipdnn_backend
)

install(
    EXPORT hipdnn_backendTargets
    FILE hipdnn_backendConfig.cmake
    NAMESPACE hipdnn::backend::
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/hipdnn_backend
)
