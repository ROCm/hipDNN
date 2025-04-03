rocroller_add_dependency(googletest VERSION 1.12.1)

add_executable(hipdnn_frontend_tests test_hipdnn_frontend_hello_world.cpp)

target_link_libraries(hipdnn_frontend_tests gtest gtest_main pthread)
target_include_directories(hipdnn_frontend_tests PRIVATE 
    ../include
)