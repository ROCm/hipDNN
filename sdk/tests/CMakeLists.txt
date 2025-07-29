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
    plugin/test_plugin_exception.cpp
    plugin/test_plugin_helpers.cpp
    plugin/test_plugin_last_error_manager.cpp
    plugin/test_plugin_data_type_helpers.cpp
    plugin/flatbuffer_utilities/test_graph_wrapper.cpp
    plugin/flatbuffer_utilities/test_engine_config_wrapper.cpp
    plugin/flatbuffer_utilities/test_engine_details_wrapper.cpp
    utilities/test_scoped_resource.cpp
    utilities/test_callback_logger.cpp
    utilities/test_migratable_memory.cpp
    utilities/test_floats.cpp
    utilities/test_shape_utilities.cpp
    utilities/test_tensor.cpp
    test_utilities/test_cpu_fp_reference_implementation.cpp
    test_utilities/test_cpu_fp_reference_validation.cpp
)

target_compile_definitions(hipdnn_sdk_tests PRIVATE
    COMPONENT_NAME="hipdnn_sdk_tests"
)

target_compile_options(hipdnn_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(hipdnn_sdk_tests 
    GTest::gtest
    pthread
    hipdnn_sdk
    hip::host
    hip::device
    spdlog_header_only
    ${CMAKE_DL_LIBS}
)

add_dependencies(check hipdnn_sdk_tests)
gtest_discover_tests(
    hipdnn_sdk_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)
