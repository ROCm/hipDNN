# tests/CMakeLists.txt
rocroller_add_dependency(googletest VERSION 1.12.1)

add_executable(test_hip_backend test_hello_world.cpp)

target_link_libraries(test_hip_backend gtest gtest_main pthread helloworld)
target_include_directories(test_hip_backend PRIVATE ${PROJECT_SOURCE_DIR}/library/include)