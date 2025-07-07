# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping SDK tests")
    return()
endif()

add_executable(hipdnn_sdk_tests
    main.cpp
    test_plugin_flatbuffer_utilities.cpp
    test_plugin_exception.cpp
    test_plugin_helpers.cpp
    test_plugin_last_error_manager.cpp
    test_plugin_data_type_helpers.cpp
)

target_link_libraries(hipdnn_sdk_tests 
    GTest::gtest
    pthread
    hipdnn_sdk
    ${CMAKE_DL_LIBS}
)