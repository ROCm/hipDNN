# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

set(CMAKE_CXX_STANDARD 20)

find_package(hip REQUIRED)

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

    add_dependencies(public_hipdnn_backend_tests ${target_name})
endfunction()

# test_good_plugin is a well-behaved plugin that returns a happy path
add_test_plugin(test_good_plugin test_good_plugin.cpp)
