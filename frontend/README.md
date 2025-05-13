# hipdnn-Frontend

hipDNN frontend provides both a C++ header-only API and a Python API to describe and execute operation graphs.  hipDNN graphs are defined by the schema found inside the [hipdnn_sdk](../sdk/include/hipdnn_sdk/data_objects/) under the data_object path.

The C++ frontend API uses the backend C API to handle requests (workspace, applicability, executing graphs etc), and the Python API is built upon the C++ frontend API.

## File Structure
Library Includes: [include](./include)

Unit tests for header classes: [tests](./tests)

Samples: [samples](../samples)

## Cmake Integration
This library generates a cmmake target called `hipdnn_frontend`.  When hipDNN is installed on your system, you should be able to use the `find_package(hipdnn_frontend)` in your cmake project.  If cmake cannot find the package, you will need to ensure the cmake modules has the install path.  By default, the install path is `/opt/rocm/lib/cmake`.

## Python API
Python bindings are currently PoC only and are subject to change.