hipdnn_add_dependency(googletest VERSION 1.12.1)

add_executable(hipdnn_backend_tests test_hello_world.cpp)

target_link_libraries(hipdnn_backend_tests gtest gtest_main pthread hipdnn_backend_private)
target_include_directories(hipdnn_backend_tests PRIVATE 
    ../src
)