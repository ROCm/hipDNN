# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Integration tests")
    return()
endif()

set(HIPDNN_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/backend/tests/descriptors
    ${CMAKE_SOURCE_DIR}/tests/backend
    ${CMAKE_SOURCE_DIR}/backend/src
    CACHE INTERNAL "Common test include directories"
)

if(HIP_DNN_BUILD_BACKEND)
    add_subdirectory(backend)
endif()