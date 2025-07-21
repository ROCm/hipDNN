# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping SDK tests")
    return()
endif()

add_compile_definitions(__HIP_PLATFORM_AMD__)
add_compile_definitions(__HIPCC__)
find_package(hip REQUIRED)
enable_language(HIP)

add_executable(hipdnn_sdk_tests
    main.cpp
    test_plugin_flatbuffer_utilities.cpp
    test_plugin_exception.cpp
    test_plugin_helpers.cpp
    test_plugin_last_error_manager.cpp
    test_plugin_data_type_helpers.cpp
    test_scoped_resource.cpp
    test_cpu_fp_reference_implementation.cpp
    test_cpu_fp_reference_validation.cpp
    test_migratable_memory.cpp
)

target_link_libraries(hipdnn_sdk_tests 
    GTest::gtest
    pthread
    hipdnn_sdk
    hip::host
    hip::device
    ${CMAKE_DL_LIBS}
)

add_dependencies(check hipdnn_sdk_tests)
gtest_discover_tests(
    hipdnn_sdk_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)
