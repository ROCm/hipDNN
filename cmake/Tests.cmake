# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

hipdnn_add_dependency(GTest v1.16.0)
include(GoogleTest)

find_package(Python3 COMPONENTS Interpreter)

# Set executable prefix based on platform
if(WIN32)
    set(EXEC_PREFIX "")
else()
    set(EXEC_PREFIX "./")
endif()

set(CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated check commands" FORCE)
set(CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated check depends" FORCE)
set(CHECK_EXECUTABLE_PATHS_GLOBAL "" CACHE INTERNAL "Accumulated check executable paths" FORCE)

# Global collections for unit tests
set(UNIT_CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated unit check commands" FORCE)
set(UNIT_CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated unit check depends" FORCE)

# Global collections for integration tests
set(INTEGRATION_CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated integration check commands" FORCE)
set(INTEGRATION_CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated integration check depends" FORCE)

function(create_test_name_validation_target)
    if(Python3_FOUND)
        # Write list of test executables with their paths to a file
        set(TEST_EXECUTABLES_FILE ${CMAKE_BINARY_DIR}/test_executables.txt)
        list(REMOVE_DUPLICATES CHECK_EXECUTABLE_PATHS_GLOBAL)
        file(WRITE ${TEST_EXECUTABLES_FILE} "")
        foreach(test_executable ${CHECK_EXECUTABLE_PATHS_GLOBAL})
            file(APPEND ${TEST_EXECUTABLES_FILE} "${test_executable}\n")
        endforeach()
        
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/test_names_validated
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/scripts/test_name_validator.py 
                    --test-executables ${TEST_EXECUTABLES_FILE}
                    --build-dir ${CMAKE_BINARY_DIR}
                    --strict
            COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/test_names_validated
            DEPENDS 
                ${CMAKE_SOURCE_DIR}/cmake/scripts/test_name_validator.py
                ${CHECK_DEPENDS_GLOBAL}
            COMMENT "Validating test names with --gtest_list_tests test collection"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            VERBATIM
        )
        
        add_custom_target(validate_test_names 
            DEPENDS ${CMAKE_BINARY_DIR}/test_names_validated)
    else()
        message(WARNING "Python3 not found. Test name validation will be skipped.")
        add_custom_target(validate_test_names
            COMMAND ${CMAKE_COMMAND} -E echo "Test name validation skipped - Python3 not found"
        )
    endif()
endfunction()

# Generic internal function to append tests to check targets
function(_append_test_to_check_target_internal TARGET WORKING_DIR TEST_TYPE STATUS_MESSAGE)
    if(STATUS_MESSAGE)
        message(STATUS "${STATUS_MESSAGE}: ${TARGET} in working directory: ${WORKING_DIR}")
    endif()
    
    if("${TEST_TYPE}" STREQUAL "UNIT")
        set(COMMAND_VAR "UNIT_CHECK_COMMAND_GLOBAL")
        set(DEPENDS_VAR "UNIT_CHECK_DEPENDS_GLOBAL")
        set(CACHE_DESC "Accumulated unit check targets")
    elseif("${TEST_TYPE}" STREQUAL "INTEGRATION")
        set(COMMAND_VAR "INTEGRATION_CHECK_COMMAND_GLOBAL")
        set(DEPENDS_VAR "INTEGRATION_CHECK_DEPENDS_GLOBAL")
        set(CACHE_DESC "Accumulated integration check targets")
    else()
        set(COMMAND_VAR "CHECK_COMMAND_GLOBAL")
        set(DEPENDS_VAR "CHECK_DEPENDS_GLOBAL")
        set(CACHE_DESC "Accumulated check targets")
    endif()
    
    set(NEW_COMMAND "")
    if("${${COMMAND_VAR}}" STREQUAL "")
        set(NEW_COMMAND cd ${WORKING_DIR} && ${TEST_ENVIRONMENT} ${EXEC_PREFIX}${TARGET})
    else()
        set(NEW_COMMAND && cd ${WORKING_DIR} && ${TEST_ENVIRONMENT} ${EXEC_PREFIX}${TARGET})
    endif()
    
    set(${COMMAND_VAR} ${${COMMAND_VAR}} ${NEW_COMMAND} CACHE INTERNAL "${CACHE_DESC}" FORCE)
    set(${DEPENDS_VAR} ${${DEPENDS_VAR}} ${TARGET} CACHE INTERNAL "Accumulated ${TEST_TYPE} check depends" FORCE)
    
    # Track the binary paths for test name validation
    file(RELATIVE_PATH REL_WORKING_DIR ${CMAKE_BINARY_DIR} ${WORKING_DIR})
    if(REL_WORKING_DIR)
        set(EXECUTABLE_PATH "${REL_WORKING_DIR}/${TARGET}")
    else()
        set(EXECUTABLE_PATH "${TARGET}")
    endif()
    set(CHECK_EXECUTABLE_PATHS_GLOBAL ${CHECK_EXECUTABLE_PATHS_GLOBAL} ${EXECUTABLE_PATH} CACHE INTERNAL "Accumulated check executable paths" FORCE)
endfunction()

# Generic internal function to finalize check targets
function(_finalize_check_target_internal TARGET_NAME COMMAND_VAR DEPENDS_VAR)
    add_custom_target(
        ${TARGET_NAME}
        COMMAND ${${COMMAND_VAR}}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${${DEPENDS_VAR}}
        VERBATIM)
    message(STATUS "Created ${TARGET_NAME} target")
endfunction()

function(finalize_custom_check_target)
    _finalize_check_target_internal("check" "CHECK_COMMAND_GLOBAL" "CHECK_DEPENDS_GLOBAL")
endfunction()

function(finalize_unit_check_target)
    _finalize_check_target_internal("unit-check" "UNIT_CHECK_COMMAND_GLOBAL" "UNIT_CHECK_DEPENDS_GLOBAL")
endfunction()

function(finalize_integration_check_target)
    _finalize_check_target_internal("integration-check" "INTEGRATION_CHECK_COMMAND_GLOBAL" "INTEGRATION_CHECK_DEPENDS_GLOBAL")
endfunction()

enable_testing() # Cmake wont discover or run tests without this line

# Add a check_ctest target which will run all tests discovered by gtest_discover_tests via ctest. 
add_custom_target(check_ctest COMMAND ${TEST_ENVIRONMENT} ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})

function(_add_gtest_target_internal APPEND_FUNCTION_SUFFIX TARGET WORKING_DIR)
    if("${APPEND_FUNCTION_SUFFIX}" STREQUAL "test")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "" "Appending check target")
    elseif("${APPEND_FUNCTION_SUFFIX}" STREQUAL "unit_test")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "UNIT" "Appending unit check target")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "" "")
    elseif("${APPEND_FUNCTION_SUFFIX}" STREQUAL "integration_test")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "INTEGRATION" "Appending integration check target")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "" "")
    else()
        message(FATAL_ERROR "Unknown test type suffix: ${APPEND_FUNCTION_SUFFIX}")
    endif()
    
    add_dependencies(check_ctest ${TARGET})
    gtest_discover_tests(
        ${TARGET}
        WORKING_DIRECTORY ${WORKING_DIR}
        DISCOVERY_MODE PRE_TEST
    )
endfunction()

# Adds a generic test target
function(add_target_to_check_targets TARGET WORKING_DIR)
    _add_gtest_target_internal(test ${TARGET} ${WORKING_DIR})
endfunction()

# Adds a unit test target
function(add_unit_test_target TARGET WORKING_DIR)
    _add_gtest_target_internal(unit_test ${TARGET} ${WORKING_DIR})
endfunction()

# Adds an integration test target
function(add_integration_test_target TARGET WORKING_DIR)
    _add_gtest_target_internal(integration_test ${TARGET} ${WORKING_DIR})
endfunction()
