# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

hipdnn_add_dependency(GTest v1.16.0)
include(GoogleTest)

# Global collections for all tests
set(CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated check commands" FORCE)
set(CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated check depends" FORCE)

# Global collections for unit tests
set(UNIT_CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated unit check commands" FORCE)
set(UNIT_CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated unit check depends" FORCE)

# Global collections for integration tests
set(INTEGRATION_CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated integration check commands" FORCE)
set(INTEGRATION_CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated integration check depends" FORCE)

function(append_test_to_check_target TARGET WORKING_DIR)
    message(STATUS "Appending check target: ${TARGET} in working directory: ${WORKING_DIR}")

    set(NEW_COMMAND "")
    if("${CHECK_COMMAND_GLOBAL}" STREQUAL "")
    set(NEW_COMMAND cd ${WORKING_DIR} && ${TEST_ENVIRONMENT} ./${TARGET})
    else()
    set(NEW_COMMAND && cd ${WORKING_DIR} && ${TEST_ENVIRONMENT} ./${TARGET})
    endif()
    set(CHECK_COMMAND_GLOBAL ${CHECK_COMMAND_GLOBAL} ${NEW_COMMAND} CACHE INTERNAL "Accumulated check targets" FORCE)
    set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} ${TARGET} CACHE INTERNAL "Accumulated check depends" FORCE)    
endfunction()

function(append_unit_test_to_check_target TARGET WORKING_DIR)
    message(STATUS "Appending unit check target: ${TARGET} in working directory: ${WORKING_DIR}")

    set(NEW_COMMAND "")
    if("${UNIT_CHECK_COMMAND_GLOBAL}" STREQUAL "")
    set(NEW_COMMAND cd ${WORKING_DIR} && ${TEST_ENVIRONMENT} ./${TARGET})
    else()
    set(NEW_COMMAND && cd ${WORKING_DIR} && ${TEST_ENVIRONMENT} ./${TARGET})
    endif()
    set(UNIT_CHECK_COMMAND_GLOBAL ${UNIT_CHECK_COMMAND_GLOBAL} ${NEW_COMMAND} CACHE INTERNAL "Accumulated unit check targets" FORCE)
    set(UNIT_CHECK_DEPENDS_GLOBAL ${UNIT_CHECK_DEPENDS_GLOBAL} ${TARGET} CACHE INTERNAL "Accumulated unit check depends" FORCE)    
    
    append_test_to_check_target(${TARGET} ${WORKING_DIR})
endfunction()

function(append_integration_test_to_check_target TARGET WORKING_DIR)
    message(STATUS "Appending integration check target: ${TARGET} in working directory: ${WORKING_DIR}")

    set(NEW_COMMAND "")
    if("${INTEGRATION_CHECK_COMMAND_GLOBAL}" STREQUAL "")
    set(NEW_COMMAND cd ${WORKING_DIR} && ${TEST_ENVIRONMENT} ./${TARGET})
    else()
    set(NEW_COMMAND && cd ${WORKING_DIR} && ${TEST_ENVIRONMENT} ./${TARGET})
    endif()
    set(INTEGRATION_CHECK_COMMAND_GLOBAL ${INTEGRATION_CHECK_COMMAND_GLOBAL} ${NEW_COMMAND} CACHE INTERNAL "Accumulated integration check targets" FORCE)
    set(INTEGRATION_CHECK_DEPENDS_GLOBAL ${INTEGRATION_CHECK_DEPENDS_GLOBAL} ${TARGET} CACHE INTERNAL "Accumulated integration check depends" FORCE)    
    
    append_test_to_check_target(${TARGET} ${WORKING_DIR})
endfunction()

function(finalize_custom_check_target)
add_custom_target(
    check
    COMMAND ${CHECK_COMMAND_GLOBAL}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${CHECK_DEPENDS_GLOBAL}
    VERBATIM)
endfunction()

function(finalize_unit_check_target)
if(NOT "${UNIT_CHECK_COMMAND_GLOBAL}" STREQUAL "")
    add_custom_target(
        unit-check
        COMMAND ${UNIT_CHECK_COMMAND_GLOBAL}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${UNIT_CHECK_DEPENDS_GLOBAL}
        VERBATIM)
    message(STATUS "Created unit-check target")
else()
    add_custom_target(
        unit-check
        COMMAND ${CMAKE_COMMAND} -E echo "No unit tests found"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        VERBATIM)
    message(STATUS "Created empty unit-check target (no unit tests found)")
endif()
endfunction()

function(finalize_integration_check_target)
if(NOT "${INTEGRATION_CHECK_COMMAND_GLOBAL}" STREQUAL "")
    add_custom_target(
        integration-check
        COMMAND ${INTEGRATION_CHECK_COMMAND_GLOBAL}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${INTEGRATION_CHECK_DEPENDS_GLOBAL}
        VERBATIM)
    message(STATUS "Created integration-check target")
else()
    add_custom_target(
        integration-check
        COMMAND ${CMAKE_COMMAND} -E echo "No integration tests found"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        VERBATIM)
    message(STATUS "Created empty integration-check target (no integration tests found)")
endif()
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

function(add_unit_test_target TARGET WORKING_DIR)
    append_unit_test_to_check_target(${TARGET} ${WORKING_DIR})
    add_dependencies(check_ctest ${TARGET})
    gtest_discover_tests(
        ${TARGET}
        WORKING_DIRECTORY ${WORKING_DIR}
        DISCOVERY_MODE PRE_TEST
    )
endfunction()

function(add_integration_test_target TARGET WORKING_DIR)
    append_integration_test_to_check_target(${TARGET} ${WORKING_DIR})
    add_dependencies(check_ctest ${TARGET})
    gtest_discover_tests(
        ${TARGET}
        WORKING_DIRECTORY ${WORKING_DIR}
        DISCOVERY_MODE PRE_TEST
    )
endfunction()
