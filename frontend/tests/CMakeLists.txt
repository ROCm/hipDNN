hipdnn_add_dependency(googletest VERSION 1.12.1)
include(GoogleTest)

add_executable(hipdnn_frontend_tests 
    test_hipdnn_frontend_hello_world.cpp
)

target_link_libraries(hipdnn_frontend_tests gtest gtest_main pthread)
target_include_directories(hipdnn_frontend_tests PRIVATE 
    ../include
)

add_dependencies(check hipdnn_frontend_tests)
gtest_discover_tests(
    hipdnn_frontend_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)
