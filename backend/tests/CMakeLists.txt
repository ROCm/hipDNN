rocroller_add_dependency(googletest VERSION 1.12.1)

add_executable(test_hipdnn_backend test_hello_world.cpp)

target_link_libraries(test_hipdnn_backend gtest gtest_main pthread hipdnn_backend)
target_include_directories(test_hipdnn_backend PRIVATE 
    ../src
)