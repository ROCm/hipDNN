# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

set(CMAKE_CXX_STANDARD 20)

find_package(hip REQUIRED)

# Define plugin names as variables
set(TEST_GOOD_PLUGIN_NAME "test_good_plugin")
set(TEST_EXECUTE_FAILS_PLUGIN_NAME "test_execute_fails_plugin")
set(TEST_NO_APPLICABLE_ENGINES_PLUGIN_NAME "test_no_applicable_engines_plugin")
set(TEST_DUPLICATE_ID_A_PLUGIN_NAME "test_duplicate_id_a_plugin")
set(TEST_DUPLICATE_ID_B_PLUGIN_NAME "test_duplicate_id_b_plugin")
set(TEST_INCOMPLETE_API_PLUGIN_NAME "test_incomplete_api_plugin")


# Function to create a test plugin with common configuration
function(add_test_plugin target_name)
    set(source_files ${ARGN})
    
    add_library(${target_name} 
        SHARED
            ${source_files}
    )

    target_compile_definitions(${target_name} 
        PRIVATE 
            COMPONENT_NAME="${target_name}"
            __HIP_PLATFORM_AMD__
    )

    target_include_directories(${target_name} 
        PUBLIC 
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${target_name} 
        PUBLIC 
            hipdnn_sdk
            hip::host
    )

    target_compile_options(${target_name} 
        PRIVATE 
            ${HIPDNN_WARNING_COMPILE_OPTIONS}
    )
endfunction()

#NOTE: If you add a new test plugin, then ensure it is listed in the frontend and backend
#      CMakeLists.txt files as a dependency, and add the plugin name define.

# test_good_plugin is a well-behaved plugin that returns a happy path
add_test_plugin(${TEST_GOOD_PLUGIN_NAME} test_good_plugin.cpp)

# test_execute_fails_plugin is a plugin that fails during execution
add_test_plugin(${TEST_EXECUTE_FAILS_PLUGIN_NAME} test_execute_fails_plugin.cpp)

# test_no_applicable_engines_plugin is a plugin that reports no applicable engines
add_test_plugin(${TEST_NO_APPLICABLE_ENGINES_PLUGIN_NAME} test_no_applicable_engines_plugin.cpp)

# test_good_plugin_duplicate_id_a/b shares an engine id with each other
add_test_plugin(${TEST_DUPLICATE_ID_A_PLUGIN_NAME} test_duplicate_id_a_plugin.cpp)
add_test_plugin(${TEST_DUPLICATE_ID_B_PLUGIN_NAME} test_duplicate_id_b_plugin.cpp)

# test_incomplete_api_plugin only implements part of the plugin API
add_test_plugin(${TEST_INCOMPLETE_API_PLUGIN_NAME} test_incomplete_api_plugin.cpp)


# Create a duplicated default plugin that gets installed to the default plugin directory
add_test_plugin(test_good_default_plugin test_good_default_plugin.cpp)

# Create the default plugin directory relative to where the .so is located (backend/src)
set(DEFAULT_PLUGIN_DIR "${CMAKE_BINARY_DIR}/backend/src/hipdnn_plugins/engines")
file(MAKE_DIRECTORY ${DEFAULT_PLUGIN_DIR})

# Install the default plugin to the default plugin directory
add_custom_command(TARGET test_good_default_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy 
        $<TARGET_FILE:test_good_default_plugin>
        ${DEFAULT_PLUGIN_DIR}/$<TARGET_FILE_NAME:test_good_default_plugin>
)

# Export plugin names to parent scope for use in test compilation
set(TEST_GOOD_PLUGIN_NAME "${TEST_GOOD_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_EXECUTE_FAILS_PLUGIN_NAME "${TEST_EXECUTE_FAILS_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_NO_APPLICABLE_ENGINES_PLUGIN_NAME "${TEST_NO_APPLICABLE_ENGINES_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_DUPLICATE_ID_A_PLUGIN_NAME "${TEST_DUPLICATE_ID_A_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_DUPLICATE_ID_B_PLUGIN_NAME "${TEST_DUPLICATE_ID_B_PLUGIN_NAME}" PARENT_SCOPE)
set(TEST_INCOMPLETE_API_PLUGIN_NAME "${TEST_INCOMPLETE_API_PLUGIN_NAME}" PARENT_SCOPE)
