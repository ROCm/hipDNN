rocroller_add_dependency(googletest VERSION 1.12.1)

add_executable(test_public_hipdnn_backend test_backend_public_functions.cpp)

target_link_libraries(test_public_hipdnn_backend gtest gtest_main pthread hipdnn_backend)
target_include_directories(test_public_hipdnn_backend PRIVATE 
    ../../backend/include
)