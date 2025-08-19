# Plugin Development

This guide provides comprehensive information for developing plugins for hipDNN. Plugins extend hipDNN's capabilities through different specialized implementations.

## Table of Contents

- [Overview](#overview)
- [Plugin Types](#plugin-types)
- [hipDNN-SDK Library](#hipdnn-sdk-library)
- [Plugin API](#plugin-api)
- [Creating a Kernel Engine Plugin](#creating-a-kernel-engine-plugin)
  - [Steps Overview](#steps-overview)
  - [Implementation Details](#implementation-details)
  - [Key Files Reference](#key-files-reference)
- [Plugin Architecture](#plugin-architecture)
- [Example: MIOpen Legacy Plugin](#example-miopen-legacy-plugin)

---

## Overview

hipDNN supports a plugin architecture that allows for modular extensions to the framework. Plugins are designed to be separate projects that extend hipDNN's capabilities without being part of the core repository. The backend discovers and manages these plugins, leveraging them for different aspects of deep learning computation. This architecture provides flexibility in implementation choices and enables optimizations for specific hardware or use cases.

## Plugin Types

hipDNN will support three types of plugins, each serving a specific purpose:

### 1. Engine Heuristic and Selection Plugins (`hipdnn_plugins/heuristics/`)
These plugins help determine the best execution strategy for a given operation or graph. They analyze the computation requirements and available resources to select optimal implementations.

### 2. Benchmarking and Tuning Plugins (`hipdnn_plugins/benchmarking/`)
These plugins focus on performance optimization by benchmarking different implementations and tuning parameters for specific hardware configurations.

### 3. Kernel Engine Plugins (`hipdnn_plugins/engines/`)
These plugins provide the actual kernel implementations for operations. They contain the compute kernels that execute on the target hardware (GPUs, accelerators, etc.).

> [!IMPORTANT]
> **Current Status**: Only kernel engine plugins are presently supported in hipDNN. The MIOpen Legacy Plugin is currently included as a reference implementation but will be migrated to its own separate project in the future. Support for engine heuristic/selection and benchmarking/tuning plugins will be added in future releases. See the [Roadmap](./Roadmap.md#plugins) for future development plans.

## hipDNN-SDK Library

The hipDNN-SDK API is a Header-Only C++ library which provides the requirements needed to create a plugin that hipDNN can consume. It includes:

- Plugin interface definitions
- Data structures for graph representation
- Utilities for serialization/deserialization
- Base classes for engine implementation

For adding new operations to the SDK (schemas, nodes, attributes), see the [How-To Guide](./HowTo.md#adding-a-new-operation-to-existing-plugins).

## Plugin API

The plugin API defines how kernel engine plugins interact with hipDNN:

- **Graph Processing**: Graphs are passed in a serialized format to plugins using FlatBuffers
- **SDK Data Objects**: Plugins use SDK data objects to deserialize and process graphs
- **Capability Reporting**: Plugins analyze graphs and report whether they can execute them
- **Execution Interface**: Plugins provide execution methods for supported operations

## Creating a Kernel Engine Plugin

This section focuses on developing kernel engine plugins, which are currently the only supported plugin type.

### Steps Overview

1. **Create Plugin Structure**
   - Create a new project/repository for your plugin
   - Implement the plugin interface defined in [`sdk/include/hipdnn_sdk/plugin/engine_plugin_api.h`](../sdk/include/hipdnn_sdk/plugin/engine_plugin_api.h)
   - See [MIOpen Legacy Plugin](../plugins/miopen_legacy_plugin/) as a reference implementation (currently included but will become a separate project)

2. **Implement Plugin API Functions**
   
   The underlying implementation below the plugin API level is entirely at the developer's discretion. While the following architectural components are recommended for code organization and maintainability; the only true requirement is to correctly implement the exported API functions defined in `engine_plugin_api.h`. However, the common architectural pattern consists of:
   - **Engine Manager**: Manages available engines and their capabilities
   - **Engine**: Implements graph execution for specific operations (each engine must have a globally unique `int64_t` ID)
   - **Execution Plans**: Define how operations are executed

3. **Build and Deploy Plugin**
   - Configure CMake to build the plugin as a shared library
   - Install to the appropriate plugin directory where hipDNN can discover it at runtime

### Implementation Details

#### Engine Manager
The Engine Manager is responsible for:
- Creating and managing engine instances
- Reporting supported operations
- Handling resource allocation
- Managing device-specific contexts

#### Engine Implementation
When implementing engines (if following this pattern):
- Each engine must have a unique `int64_t` identifier within the plugin
- Implement the `execute()` method for graph execution
- Provide `get_supported_operations()` to report capabilities
- Handle operation-specific kernel launches
- Manage memory transfers and synchronization

> [!TIP]
> An engine ID is an integer unique to all loaded plugins. These IDs are used by the backend to identify and select specific engines for execution. You may want to reference other loaded plugins to accrue a set of unused engine IDs.

#### Execution Plans
Execution plans for kernel engines:
- Map hipDNN operations to backend-specific kernel implementations
- Define memory layouts and data transformations
- Specify kernel launch configurations
- Handle device-specific optimizations

### Key Files Reference

- **Plugin API Interface**: [`sdk/include/hipdnn_sdk/plugin/engine_plugin_api.h`](../sdk/include/hipdnn_sdk/plugin/engine_plugin_api.h)
- **Example Plugin Implementation**: [`plugins/miopen_legacy_plugin/miopen_legacy_plugin.cpp`](../plugins/miopen_legacy_plugin/miopen_legacy_plugin.cpp)
- **Example Engine Manager**: [`plugins/miopen_legacy_plugin/engine_manager.cpp`](../plugins/miopen_legacy_plugin/engine_manager.cpp)
- **Example Engine Implementation**: [`plugins/miopen_legacy_plugin/engines/miopen_engine.cpp`](../plugins/miopen_legacy_plugin/engines/miopen_engine.cpp)

## Plugin Architecture

### Directory Structure for Kernel Engine Plugins

Your plugin should be structured as an independent project:

```
your_kernel_plugin_project/
├── CMakeLists.txt
├── your_plugin.cpp           # Main plugin entry point
├── engine_manager.cpp        # Engine management
├── engines/
│   ├── engine_interface.hpp  # Engine interface
│   ├── your_engine.cpp       # Engine implementation
│   └── plans/                # Internal plans
├── tests/                    # Plugin-specific tests
└── integration_tests/        # End-to-end integration tests
```

### Build Configuration
Your plugin's CMakeLists.txt should:
- Build as a shared library
- Link against hipDNN SDK
- Set appropriate install paths
- Link to required compute libraries (ie. HIP)

### Plugin Loading
Plugins are discovered and loaded from:
- Default path: `hipdnn_plugins/<plugin-type>/` relative to the backend library
- Custom paths can be configured using environment variables
- See [Environment Configuration](./Environment.md) for details

## How to Test Plugins

Testing is crucial for ensuring plugin reliability and correctness. Plugins should include both unit tests and integration tests to validate their functionality.

### Test Structure

Following the [Testing Strategy](./testing/TestingStrategy.md), plugins should organize tests as follows:

```
your_kernel_plugin_project/
├── tests/                    # Unit tests
│   ├── test_engine.cpp
│   ├── test_kernels.cpp
│   └── test_utilities.cpp
└── integration_tests/        # End-to-end integration tests
    ├── operation1_test.cpp
    └── operation2_test.cpp
```

### Unit Tests

Unit tests focus on the internal implementation of your plugin components:

- **Location**: `plugins/<plugin_name>/tests/`
- **Purpose**: Test individual components in isolation (engines, utilities, kernel logic)
- **Requirements**:
  - Must be fast-running
  - GPU operations must be marked with `SKIP_IF_NO_DEVICE()` macro
  - Use mocking/stubbing for dependencies where appropriate
  - Should work on both Windows and Linux

### Integration Tests

Integration tests validate end-to-end functionality of your plugin:

- **Location**: `plugins/<plugin_name>/integration_tests/`
- **Purpose**: Validate correctness of graph execution and accuracy of results
- **Requirements**:
  - Test complete operation graphs
  - Validate against reference implementations
  - Test different data types, layouts, dimensions, and edge-cases for each
  - Enable tests for all supported ASICs
  - GPU typically required for meaningful validation

For a comprehensive example of an integration test, see: [`plugins/miopen_legacy_plugin/integration_tests/batchnorm_fwd_inference_integration_test.cpp`](../plugins/miopen_legacy_plugin/integration_tests/batchnorm_fwd_inference_integration_test.cpp)

Moreover, see our [general testing requirements](./testing/TestingStrategy.md#general-testing-requirements).

## Example: MIOpen Legacy Plugin

The MIOpen Legacy Plugin demonstrates a complete kernel engine plugin implementation:

### Structure
- **Main Plugin**: [`miopen_legacy_plugin.cpp`](../plugins/miopen_legacy_plugin/miopen_legacy_plugin.cpp) - Entry point and plugin registration
- **Engine Manager**: [`engine_manager.cpp`](../plugins/miopen_legacy_plugin/engine_manager.cpp) - Manages MIOpen engines
- **MIOpen Engine**: [`engines/miopen_engine.cpp`](../plugins/miopen_legacy_plugin/engines/miopen_engine.cpp) - Implements graph execution using MIOpen kernels

### Integration Points
- Uses MIOpen handles for GPU context management
- Converts hipDNN tensors to MIOpen tensor descriptors
- Launches MIOpen kernels for computation
- Handles stream synchronization

---

## Best Practices for Kernel Engine Plugins

1. **Kernel Management**: Organize kernels by operation type for maintainability
2. **Memory Management**: Efficiently manage device memory allocations and transfers
3. **Stream Handling**: Properly manage compute streams for asynchronous execution
4. **Error Handling**: Validate inputs and provide meaningful error messages
5. **Testing**: Include unit tests for kernels and integration tests for operations
6. **Performance**: Profile kernels and optimize for target hardware
7. **Documentation**: Document supported operations, hardware requirements, and limitations

## Future Plugin Types

While kernel engine plugins are the focus of current development, future releases will support:
- **Engine heuristic plugins** for intelligent kernel selection
- **Benchmarking plugins** for automated performance tuning

These plugin types will follow similar development patterns but with specialized interfaces tailored to their specific purposes.
