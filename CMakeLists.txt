# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.27...3.29)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/ClangToolChain.cmake" CACHE STRING "Toolchain file")
set(HIPDNN_VERSION "0.0.1")
project(
    hipDNN
    VERSION ${HIPDNN_VERSION}
    LANGUAGES CXX)
    
set(CMAKE_CXX_STANDARD 17)

list(
    APPEND
    CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${ROCM_PATH}/lib/cmake/hip
    /opt/rocm/lib/cmake/hip
    ${HIP_DIR}/cmake
)


include(cmake/Dependencies.cmake)
include(cmake/ClangCheck.cmake)
hipdnn_add_dependency(
    ROCmCMakeBuildTools
    VERSION 0.11.0
    FIND_PACKAGE_ARGS NO_DEFAULT_PATH # PATHS "${ROCM_PATH}"
)

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMClients)

option(HIP_DNN_BUILD_BACKEND "Build the backend code" ON)
option(HIP_DNN_BUILD_FRONTEND "Build the frontend code" ON)

option(CODE_COVERAGE "Build with code coverage flags" OFF)

enable_testing() # Cmake wont discover or run tests without this line

# Add a check target which will run all tests discovered by gtest_discover_tests via ctest. 
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})

# Add global compile options
# fPIC: generates position-independent code,
set(EXTRA_COMPILE_OPTIONS)
set(EXTRA_LINK_OPTIONS)
if(CODE_COVERAGE)
    list(PREPEND EXTRA_COMPILE_OPTIONS -fprofile-instr-generate
         -fcoverage-mapping
    )
    list(PREPEND EXTRA_LINK_OPTIONS -fprofile-instr-generate -fcoverage-mapping)
endif()
add_compile_options(-fPIC ${EXTRA_COMPILE_OPTIONS})
add_link_options(${EXTRA_LINK_OPTIONS})

set(HIP_DNN_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sdk/include")
add_subdirectory(sdk)

if(HIP_DNN_BUILD_BACKEND)
    add_subdirectory(backend)
endif()

if(HIP_DNN_BUILD_FRONTEND)
    add_subdirectory(frontend)
endif()

add_subdirectory(tests)

if(CODE_COVERAGE)
    set(CODE_COVERAGE_IGNORE_REGEX "\(.*deps.*\)|\(.*tests.*\)")
    add_custom_target(
        code_coverage
        COMMAND find . -name "*.profraw" -print0 | xargs -0 /opt/rocm/llvm/bin/llvm-profdata merge -sparse -o ./hipdnn.profdata
        COMMAND /opt/rocm/llvm/bin/llvm-cov report -object ./backend/src/libhipdnn_backend.so -object ./frontend/tests/hipdnn_frontend_tests -instr-profile=./hipdnn.profdata -ignore-filename-regex=\"${CODE_COVERAGE_IGNORE_REGEX}\" > ./code_cov_hipdnn.report
        COMMAND /opt/rocm/llvm/bin/llvm-cov show -Xdemangler=/opt/rocm/llvm/bin/llvm-cxxfilt -object ./backend/src/libhipdnn_backend.so -object ./frontend/tests/hipdnn_frontend_tests -instr-profile=./hipdnn.profdata -ignore-filename-regex=\"${CODE_COVERAGE_IGNORE_REGEX}\" > ./code_cov_hipdnn.txt
        DEPENDS check
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating code coverage report"
    )
endif()
