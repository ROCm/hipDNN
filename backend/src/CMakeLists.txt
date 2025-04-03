add_library(hipdnn_backend 
    hello_world.cpp
    hipdnn_backend_api.cpp)
target_include_directories(hipdnn_backend PUBLIC ../include)