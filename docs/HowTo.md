# hipDNN How-To Guide

This guide provides practical information for both using hipDNN components and extending the framework with new functionality.

## Table of Contents

- [Consuming hipDNN](#consuming-hipdnn)
  - [Using the Frontend](#using-the-frontend)
  - [Using the Backend](#using-the-backend)
  - [Using the SDK](#using-the-sdk)
  - [CMake Integration](#cmake-integration)
  - [Logging Setup](#logging-setup)
  - [Working with Schemas](#working-with-schemas)
- [Extending hipDNN](#extending-hipdnn)
  - [Adding a New Plugin](#adding-a-new-plugin)
  - [Adding a New Operation](#adding-a-new-operation)
  - [Development Workflow](#development-workflow)

---

## Consuming hipDNN

This section covers how to use the various components of hipDNN in your applications.

### Using the Frontend

The hipDNN frontend provides a C++ header-only API to describe and execute operation graphs. The C++ frontend API uses the backend C API to handle requests (workspace, applicability, executing graphs, etc.).

#### Key Features
- Header-only C++ API
- High-level interface for graph construction
- Automatic backend integration

#### File Structure
- Library includes: [`frontend/include/`](../frontend/include/)
- Unit tests: [`frontend/tests/`](../frontend/tests/)
- Samples: [`samples/`](../samples/)

### Using the Backend

The hipDNN backend is a shared library that provides a C API to describe and execute operation graphs. It is the main component of hipDNN as it provides the capability to use plugins which can solve graphs.

#### Key Features
- C API for cross-language compatibility
- Plugin management and loading
- Graph execution orchestration

#### File Structure
- Public includes: [`backend/include/`](../backend/include/)
- Public API tests: [`tests/backend/`](../tests/backend/)

### Using the SDK

The hipDNN SDK is a header-only C++ library that provides the requirements needed to create plugins that hipDNN can consume, alongside other utilities for hipDNN such as logging and schema definitions.

#### Key Features
- Plugin interface definitions (`plugin_api.h`)
- Schema-based data objects using FlatBuffers
- Test utilities for hipDNN components
- Logging utilities using spdlog

### CMake Integration

hipDNN components can be easily integrated into your CMake projects using the installed package files.

#### Frontend Integration
```cmake
find_package(hipdnn_frontend REQUIRED)
target_link_libraries(your_target PRIVATE hipdnn::frontend)
```

#### Backend Integration
```cmake
find_package(hipdnn_backend REQUIRED)
target_link_libraries(your_target PRIVATE hipdnn::backend)
```

#### SDK Integration
```cmake
find_package(hipdnn_sdk REQUIRED)
target_link_libraries(your_plugin PRIVATE hipdnn::sdk)
```

> [!NOTE]
> If CMake cannot find the packages after installation, ensure your `CMAKE_PREFIX_PATH` includes the install location. By default on Linux systems, hipDNN CMake files are installed to `/opt/rocm/lib/cmake`.

### Logging Setup

hipDNN uses the spdlog header-only library for logging. See the [Environment docs](./Environment.md#logging-configuration) for further details.

### Working with Schemas

hipDNN uses FlatBuffers for schema-based data objects to describe graphs and operations.

#### Key Concepts
- Graphs and operations are defined using `.fbs` schema files
- Attributes marked as `long` types in graphs are foreign keys to the `uid` in `tensor_attributes`
- Schema files are located in [`sdk/schemas/`](../sdk/schemas/)
---

## Extending hipDNN

This section covers how to extend hipDNN with new functionality.

### Adding a New Plugin

Plugins extend hipDNN to support new or additional implementations of kernel engines, benchmarking, and heuristics. For comprehensive guidance on plugin development, including architecture details, implementation steps, and examples, see the [Plugin Development Guide](./PluginDevelopment.md).

### Adding a New Operation

Adding a new operation requires coordinated changes across multiple components. Here's the complete workflow:

#### Prerequisites

When adding a completely new operation type (not currently supported in hipDNN), you'll need to:

1. Define the operation in the SDK schemas
2. Create frontend classes
3. Implement the operation in target plugins

#### SDK Schema Changes

If the operation is new to hipDNN, start by defining its data structures:

1. **Create Attribute Schema**
   - Add a new `.fbs` file in [`sdk/schemas/`](../sdk/schemas/)
   - Define the operation's attributes (parameters, configurations)
   - Example: [`sdk/schemas/batchnorm_attributes.fbs`](../sdk/schemas/batchnorm_attributes.fbs)

2. **Update Graph Schema**
   - Modify [`sdk/schemas/graph.fbs`](../sdk/schemas/graph.fbs)
   - Add your new attributes to the `NodeAttributes` union
   - Include your schema file

Example:
```flatbuffers
include "your_operation_attributes.fbs";

union NodeAttributes {
    BatchnormInferenceAttributes,
    PointwiseAttributes,
    ...
    YourOperationAttributes  // Add your new operation
}
```

After updating FlatBuffer schemas, regenerate the C++ headers:

```bash
make generate_hipdnn_sdk_headers
```

#### Frontend Implementation

Create C++ classes to expose the operation to users:

1. **Create Node Class**
   - Add header file in [`frontend/include/hipdnn_frontend/node/`](../frontend/include/hipdnn_frontend/node/)
   - Inherit from the base `Node` class
   - Example: [`frontend/include/hipdnn_frontend/node/batchnorm_node.hpp`](../frontend/include/hipdnn_frontend/node/batchnorm_node.hpp)

2. **Create Attribute Classes**
   - Add corresponding attribute classes in [`frontend/include/hipdnn_frontend/attributes/`](../frontend/include/hipdnn_frontend/attributes/)
   - These wrap the FlatBuffer-generated structures

3. **Update Frontend Tests**
   - Add tests for your new node and attributes
   - See examples in [`frontend/tests/`](../frontend/tests/)

#### Plugin Integration

Refer to the [Plugin Development Guide](./PluginDevelopment.md) to implement the operation execution in target plugins. 

---

## Development Workflow

### Typical Development Flow

1. **For New Operations**:
   ```
   SDK Schema → Frontend Classes → Plugin Implementation → Tests
   ```

2. **For Existing Operations in New Plugins**:
   ```
   Plugin Implementation → Integration Tests
   ```

### Building and Testing

1. **Rebuild SDK** after schema changes:
   ```bash
   cd build
   ninja hipdnn_sdk
   ```

2. **Rebuild Frontend** after adding new classes:
   ```bash
   ninja hipdnn_frontend
   ```

3. **Test Your Implementation**:
   - Unit tests for individual components
   - Integration tests for end-to-end functionality
   - See [Testing Documentation](../tests/README.md)

### Important Considerations

- **Backward Compatibility**: Ensure schema changes don't break existing operations
- **Plugin Discovery**: For example, engine plugins are loaded from `hipdnn_plugins/engines/` relative to the backend library
- **Error Handling**: Implement proper error reporting through the plugin API
- **Performance**: Optimization is critical for facilitating plugin adoption

### Debugging Tips

- Enable logging with environment variables (see [Environment Configuration](./Environment.md))
- Use integration tests to verify operation behavior
- Check plugin loading with `HIPDNN_LOG_LEVEL=info`
- For plugin issues, check the default plugin path or use custom paths with `hipdnnSetEnginePluginPaths_ext`

---

## Additional Resources

- [Plugin Development Guide](./PluginDevelopment.md) - Detailed plugin architecture
- [Design Overview](./Design.md) - hipDNN architecture and design principles
- [Building Guide](./Building.md) - Build configuration and options
- [Environment Configuration](./Environment.md) - Runtime configuration and logging
- [Testing Documentation](../tests/README.md) - Testing guidelines
