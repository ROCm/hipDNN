# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(CLANG_FORMAT_PRUNE -path "./build" -prune -o -path "./sdk/include" -prune -o)
set(CLANG_FORMAT_BINARY /opt/rocm/llvm/bin/clang-format)
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