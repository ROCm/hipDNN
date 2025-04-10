# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(CLANG_FORMAT_PRUNE -path "./build" -prune -o -path "./sdk/include" -prune -o)
add_custom_target(
    check_format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|c\\|h\\)" -exec clang-format-12 --dry-run --Werror {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)

add_custom_target(
    format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|c\\|h\\)" -exec clang-format-12 --verbose -i {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)