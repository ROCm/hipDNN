# library/CMakeLists.txt
add_library(helloworld src/hello_world.cpp)
target_include_directories(helloworld PUBLIC include)