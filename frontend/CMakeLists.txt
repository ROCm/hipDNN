# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(HIP_DNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Path to the HIP DNN frontend include directory")

option(HIP_DNN_FRONTEND_BUILD_PYTHON_BINDINGS "Defines if python bindings are built or not." OFF)

if (HIP_DNN_FRONTEND_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
add_subdirectory(tests)
