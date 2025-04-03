add_library(hipdnn_backend_impl
    hello_world.cpp
    )
target_include_directories(hipdnn_backend_impl PUBLIC ../include)


add_library(hipdnn_backend SHARED 
    hipdnn_backend_api.cpp)

target_include_directories(hipdnn_backend PUBLIC ../include)
target_link_libraries(hipdnn_backend PRIVATE hipdnn_backend_impl)
