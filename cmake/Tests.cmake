# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

hipdnn_add_dependency(GTest v1.16.0)
include(GoogleTest)

set(CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated check commands" FORCE)
set(CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated check depends" FORCE)
function(append_test_to_check_target TARGET WORKING_DIR)
    message(STATUS "Appending check target: ${TARGET} in working directory: ${WORKING_DIR}")

    set(NEW_COMMAND "")
    if("${CHECK_COMMAND_GLOBAL}" STREQUAL "")
    set(NEW_COMMAND cd ${WORKING_DIR} && ./${TARGET})
    else()
    set(NEW_COMMAND && cd ${WORKING_DIR} && ./${TARGET})
    endif()
    set(CHECK_COMMAND_GLOBAL ${CHECK_COMMAND_GLOBAL} ${NEW_COMMAND} CACHE INTERNAL "Accumulated check targets" FORCE)
    set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} ${TARGET} CACHE INTERNAL "Accumulated check depends" FORCE)    
endfunction()

function(finalize_custom_check_target)
add_custom_target(
    check
    COMMAND ${CHECK_COMMAND_GLOBAL}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${CHECK_DEPENDS_GLOBAL}
    VERBATIM)
endfunction()

enable_testing() # Cmake wont discover or run tests without this line

# Add a check_ctest target which will run all tests discovered by gtest_discover_tests via ctest. 
add_custom_target(check_ctest COMMAND ${TEST_ENVIRONMENT} ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})

function(add_target_to_check_targets TARGET WORKING_DIR)
    append_test_to_check_target(${TARGET} ${WORKING_DIR})
    add_dependencies(check_ctest ${TARGET})
    gtest_discover_tests(
        ${TARGET}
        WORKING_DIRECTORY ${WORKING_DIR}
        DISCOVERY_MODE PRE_TEST
    )
endfunction()