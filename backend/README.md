# hipDNN-Backend

This is the installable backend for the hipDNN project.  It is a shared library that provides a C API to describe and execute operation graphs.  This library is the main component of hipDNN as it provides the capability to use plugins which can solve graphs.  hipDNN graphs are defined by the schema found inside the hipdnn SDK.  The hipDNN kernel implementations will live inside plugins.

## File Structure
Public Includes: [include](./include)

Private Implemenation source: [src](./src)

Unit tests for private source: [Unit tests](./tests)

Tests for the public API: [Public Api Tests](../tests/backend/)

## Cmake Integration
- in the future there will be a hipdnn_backend target which can be used with cmake's `find_package`command.