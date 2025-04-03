add_library(hipdnn_backend_impl
    hello_world.cpp
    )
target_include_directories(hipdnn_backend_impl PUBLIC ../include)


add_library(hipdnn_backend SHARED hipdnn_backend_api.cpp)
target_include_directories(hipdnn_backend PUBLIC ../include ${PROJECT_BINARY_DIR}/backend/include/)
target_link_libraries(hipdnn_backend PRIVATE hipdnn_backend_impl)
target_link_libraries(hipdnn_backend PRIVATE "-Xlinker --exclude-libs=ALL" ) # HIDE symbols from linked static libs

include( GenerateExportHeader )
generate_export_header( hipdnn_backend 
                        EXPORT_FILE_NAME 
                        ${PROJECT_BINARY_DIR}/backend/include/hipdnn_backend_export.h )