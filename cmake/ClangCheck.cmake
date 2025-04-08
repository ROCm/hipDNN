# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# use cmake glob_recurse to find all source files that are not in the build folder
# this isnt the most ideal way to do this, but it works for now.  I attempted to use a find . command but kept hitting issues
# the find command I tried:  COMMAND find ${CMAKE_SOURCE_DIR}/* -regex '.*\.\(cpp\|hpp\|c\|h\)' -not -path \"${CMAKE_BINARY_DIR}/*\" -exec clang-format-12 --dry-run --Werror {} +
file(GLOB_RECURSE SRC_FILES
    CONFIGURE_DEPENDS 
    *.c *.h *.cpp *.hpp
)

list(FILTER SRC_FILES EXCLUDE REGEX "/build*/|/sdk/include*/")

add_custom_target(
    check_format
    COMMAND clang-format-12 --dry-run --Werror ${SRC_FILES}
)

add_custom_target(
    format
    COMMAND clang-format-12 --verbose -i ${SRC_FILES}
)