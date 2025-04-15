# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_program(BASH_EXECUTABLE bash)
if(NOT BASH_EXECUTABLE)
    message(FATAL_ERROR "Bash executable not found.")
endif()

add_test(
    NAME pythonapi_tests
    COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_wrapper.sh ${CMAKE_CURRENT_SOURCE_DIR}
)