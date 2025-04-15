find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(hipdnn
    MODULE hipdnn.cpp
           attributes.cpp
           types.cpp
    WITH_SOABI)
target_link_libraries(hipdnn PRIVATE pybind11::headers)

target_compile_definitions(hipdnn PRIVATE)

target_include_directories(hipdnn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../frontend/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../sdk/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../build/_deps/flatbuffers-src/include)

add_subdirectory(tests)