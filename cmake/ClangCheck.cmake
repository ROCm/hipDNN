# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(CLANG_FORMAT_PRUNE -path "./build" -prune -o -path "./sdk/include/hipdnn_sdk/data_objects" -prune -o)

get_filename_component(COMPILER_PATH "${CMAKE_CXX_COMPILER}" PATH)
find_program(CLANG_FORMAT_BINARY
    NAMES
        clang-format-18
        clang-format
    PATHS
        /usr/bin
        /opt/rocm/llvm/bin
        ${COMPILER_PATH}
)

if(NOT CLANG_FORMAT_BINARY)
    message(FATAL_ERROR "clang-format not found in PATH, /opt/rocm/llvm/bin, or compiler directory")
endif()

# Check clang-format version
set(EXPECTED_CLANG_FORMAT_VERSION "18")
execute_process(
    COMMAND ${CLANG_FORMAT_BINARY} --version
    OUTPUT_VARIABLE CLANG_FORMAT_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract version number from output (format: "clang-format version X.Y.Z...")
if(CLANG_FORMAT_VERSION_OUTPUT MATCHES "clang-format version ([0-9]+)\\.")
    set(CLANG_FORMAT_MAJOR_VERSION "${CMAKE_MATCH_1}")
    if(NOT CLANG_FORMAT_MAJOR_VERSION STREQUAL EXPECTED_CLANG_FORMAT_VERSION)
        message(WARNING 
            "clang-format version mismatch!\n"
            "  Expected: ${EXPECTED_CLANG_FORMAT_VERSION}\n"
            "  Found: ${CLANG_FORMAT_MAJOR_VERSION}\n"
            "  Full version string: ${CLANG_FORMAT_VERSION_OUTPUT}\n"
            "  This may lead to inconsistent formatting.")
    else()
        message(STATUS "Found clang-format version ${CLANG_FORMAT_MAJOR_VERSION} at ${CLANG_FORMAT_BINARY}")
    endif()
else()
    message(WARNING "Could not determine clang-format version from: ${CLANG_FORMAT_VERSION_OUTPUT}")
endif()

add_custom_target(
    check_format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|c\\|h\\)" -exec ${CLANG_FORMAT_BINARY} --dry-run --Werror {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)

add_custom_target(
    format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|c\\|h\\)" -exec ${CLANG_FORMAT_BINARY} --verbose -i {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)
