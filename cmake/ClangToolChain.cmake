# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Ensure the /opt/rocm/llvm/bin path exists before setting the compilers

if(UNIX)
    set(ROCM_LLVM_BIN_DIR /opt/rocm/llvm/bin)

    if(EXISTS ${ROCM_LLVM_BIN_DIR})
        # Set the C and C++ compilers to clang and clang++ with a specific directory hint
        set(CMAKE_C_COMPILER ${ROCM_LLVM_BIN_DIR}/clang)
        set(CMAKE_CXX_COMPILER ${ROCM_LLVM_BIN_DIR}/clang++)
    else()
        message(WARNING "The directory /opt/rocm/llvm/bin does not exist. Cannot auto select clang compilers.")
    endif()
endif()