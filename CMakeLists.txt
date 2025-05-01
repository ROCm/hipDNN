# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/ClangToolChain.cmake" CACHE STRING "Toolchain file")
set(HIPDNN_VERSION "0.0.1")

if(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
endif()

if(NOT WIN32)
    if(CMAKE_GENERATOR STREQUAL "Ninja")
        message(STATUS "The CMake generator is Ninja.")
    else()
        message(WARNING "The CMake generator is not Ninja. Ninja is preferred over Make.")
    endif()
endif()

project(
    hipDNN
    VERSION ${HIPDNN_VERSION}
    LANGUAGES CXX)
    
set(CMAKE_CXX_STANDARD 20)

option(HIP_DNN_BUILD_BACKEND "Build the backend code" ON)
option(HIP_DNN_BUILD_FRONTEND "Build the frontend code" ON)
option(HIP_DNN_SKIP_TESTS "Skips building all tests" OFF)

option(CODE_COVERAGE "Build with code coverage flags" OFF)
option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)

list(
    APPEND
    CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${ROCM_PATH}/lib/cmake/hip
    /opt/rocm/lib/cmake/hip
    ${HIP_DIR}/cmake
)

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files; remove when fixed.
#       Added to prevent issues with not finding hip cmake.
list(
    APPEND
    CMAKE_PREFIX_PATH
    ${ROCM_PATH}/llvm
    ${ROCM_PATH}
    ${ROCM_PATH}/hip
    /opt/rocm/llvm
    /opt/rocm
    /opt/rocm/hip 
)

include(cmake/Dependencies.cmake)
include(cmake/ClangCheck.cmake)
include(cmake/ClangTidy.cmake)

hipdnn_add_dependency(GTest v1.16.0)

enable_testing() # Cmake wont discover or run tests without this line

# Add a check target which will run all tests discovered by gtest_discover_tests via ctest. 
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})

# Add global compile options
set(EXTRA_COMPILE_OPTIONS)
set(EXTRA_LINK_OPTIONS)
if(CODE_COVERAGE)
    list(PREPEND EXTRA_COMPILE_OPTIONS 
        -fprofile-instr-generate
        -fcoverage-mapping
    )
    list(PREPEND EXTRA_LINK_OPTIONS 
        -fprofile-instr-generate 
        -fcoverage-mapping)
endif()

list(PREPEND HIPDNN_WARNING_COMPILE_OPTIONS
    -Werror                                  # Treat all warnings as errors
    -Wall                                    # Enable most common warnings
    -Wextra                                  # Enable additional warnings not covered by -Wall
    -Wpedantic                               # Enforce strict ISO C++ compliance
    -Wshadow                                 # Warn about variable shadowing
    -Wnon-virtual-dtor                       # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast                         # Warn about C-style casts
    -Wcast-align                             # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual                     # Warn if a base class function is hidden by a derived class function with the same name
    -Wconversion                             # Warn about implicit type conversions that may alter a value
    -Wsign-conversion                        # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference                       # Warn about dereferencing null pointers
    -Wdouble-promotion                       # Warn when a float is implicitly promoted to a double
    -Wformat=2                               # Enable stricter format string checks
    -Winit-self                              # Warn about variables initialized with itself
    -Wunreachable-code                       # Warn about unreachable code
    -Wno-error=unused-command-line-argument  # Disable error for unused command line arguments
)

add_compile_options(
    -fPIC # Position Independent Code
    ${EXTRA_COMPILE_OPTIONS})
add_link_options(${EXTRA_LINK_OPTIONS})

add_subdirectory(sdk)

if(HIP_DNN_BUILD_BACKEND)
    add_subdirectory(backend)
endif()

if(HIP_DNN_BUILD_FRONTEND)
    add_subdirectory(frontend)
endif()

add_subdirectory(tests)

if(CODE_COVERAGE)
    set(CODE_COVERAGE_IGNORE_REGEX "\(.*deps.*\)|\(.*tests.*\)|\(.*sdk/include.*\)")
    set(CODE_COVERAGE_OBJECTS -object ./backend/src/libhipdnn_backend.so -object ./backend/tests/hipdnn_backend_tests -object ./frontend/tests/hipdnn_frontend_tests -object ./tests/backend/public_hipdnn_backend_tests)
    add_custom_target(
        code_coverage
        COMMAND cd backend/tests && ./hipdnn_backend_tests
        COMMAND cd frontend/tests && ./hipdnn_frontend_tests
        COMMAND cd tests/backend && ./public_hipdnn_backend_tests
        COMMAND find . -name "*.profraw" -print0 | xargs -0 /opt/rocm/llvm/bin/llvm-profdata merge -sparse -o ./hipdnn.profdata
        COMMAND /opt/rocm/llvm/bin/llvm-cov report ${CODE_COVERAGE_OBJECTS} -instr-profile=./hipdnn.profdata -ignore-filename-regex=\"${CODE_COVERAGE_IGNORE_REGEX}\" > ./code_cov_hipdnn.report
        COMMAND /opt/rocm/llvm/bin/llvm-cov show -Xdemangler=/opt/rocm/llvm/bin/llvm-cxxfilt ${CODE_COVERAGE_OBJECTS} -instr-profile=./hipdnn.profdata -ignore-filename-regex=\"${CODE_COVERAGE_IGNORE_REGEX}\" --format=html --output-dir=./hipdnn_coverage_html
        # We cant depend on the make check target due to the fact that ctest runs each test in a separate process
        # this causes the .profraw file to get rewritten every time a test executes.  This means we only get coverage
        # for the test that runs last.  The current workaround is to simply invoke the gtests directly.
        # This is not ideal but it works for now.
        #DEPENDS check
        DEPENDS hipdnn_backend_tests hipdnn_frontend_tests public_hipdnn_backend_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating code coverage report"
    )
endif()
