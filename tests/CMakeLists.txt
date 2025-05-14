# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Integration tests")
    return()
endif()

if(HIP_DNN_BUILD_BACKEND)
    add_subdirectory(backend)
endif()