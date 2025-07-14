# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping SDK tests")
    return()
endif()

add_executable(hipdnn_sdk_tests
    main.cpp
    plugin/test_plugin_exception.cpp
    plugin/test_plugin_helpers.cpp
    plugin/test_plugin_last_error_manager.cpp
    plugin/test_plugin_data_type_helpers.cpp
    plugin/flatbuffer_utilities/test_graph_wrapper.cpp
    plugin/flatbuffer_utilities/test_engine_config_wrapper.cpp
    plugin/flatbuffer_utilities/test_engine_details_wrapper.cpp
)

target_link_libraries(hipdnn_sdk_tests 
    GTest::gtest
    pthread
    hipdnn_sdk
    ${CMAKE_DL_LIBS}
)