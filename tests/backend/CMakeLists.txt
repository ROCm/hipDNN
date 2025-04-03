hipdnn_add_dependency(googletest VERSION 1.12.1)

add_executable(public_hipdnn_backend_tests test_backend_public_functions.cpp)

target_link_libraries(public_hipdnn_backend_tests gtest gtest_main pthread hipdnn_backend)
target_include_directories(public_hipdnn_backend_tests PRIVATE 
    ../../backend/include
)