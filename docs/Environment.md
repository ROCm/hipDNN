# hipDNN Environment Configuration

This document describes the environment variables and runtime configuration options for hipDNN.

## Table of Contents

- [Environment Variables](#environment-variables)
  - [Logging Configuration](#logging-configuration)
  - [MIOpen Plugin Logging](#miopen-plugin-logging)
- [Plugin Loading](#plugin-loading)
  - [Default Plugin Loading](#default-plugin-loading)
  - [Custom Plugin Paths](#custom-plugin-paths)
- [Error Handling](#error-handling)

---

## Environment Variables

### Logging Configuration

hipDNN provides two environment variables to control logging behavior:

#### HIPDNN_LOG_LEVEL

Controls the verbosity level of logging output.

| Level | Description |
|-------|-------------|
| `off` | No logging (default) |
| `info` | Informational messages |
| `warn` | Warning messages |
| `error` | Error messages |
| `fatal` | Fatal error messages |

**Example:**
```bash
export HIPDNN_LOG_LEVEL=info
```

#### HIPDNN_LOG_FILE

Specifies the file path where logs will be **appended**. If not set, logs are written to `stderr`.

**Example:**
```bash
export HIPDNN_LOG_FILE=/path/to/hipdnn.log
```

### Frontend and Plugin Logging

The frontend and plugins can be configured to use the same logging destination as the backend, which is lazy-initialized automatically:

1. Initialize logging using the `initialize_callback_logging` function
2. Pass `hipdnnLoggingCallback_ext` as the callback function (accessible via plugin API or backend header)
3. This ensures all components log to the same destination

### MIOpen Plugin Logging

> [!TIP]
> When using the MIOpen legacy plugin, you can use MIOpen-specific environment variables to control the underlying library's logging behavior.

For more details about MIOpen logging, see the latest [MIOpen Debug and Logging documentation](https://rocm.docs.amd.com/projects/MIOpen/en/develop/how-to/debug-log.html). All MIOpen environment variables remain compatible with hipDNN's MIOpen legacy plugin.

---

## Plugin Loading

hipDNN supports dynamic plugin loading with configurable search paths.

### Default Plugin Loading

By default, hipDNN loads plugins from:
```
./hipdnn_plugins/plugin_type/plugins
```

This path is relative to the backend shared library location, typically:
```
/opt/rocm/lib/hipdnn/
```

**Default structure example:**
```
/opt/rocm/lib/hipdnn/
└── hipdnn_plugins/
    └── engines/
        └── plugins/
            ├── miopen_legacy_plugin.so
            └── other_plugin.so
```

### Custom Plugin Paths

Prior to creating a hipDNN handle, you can specify custom plugin paths using the `hipdnnSetEnginePluginPaths_ext` function:

```c
hipdnnStatus_t hipdnnSetEnginePluginPaths_ext(
    size_t num_paths,
    const char* const* plugin_paths,
    hipdnnPluginLoadingMode_ext_t loading_mode
);
```

#### Path Resolution

Custom paths can be:
- **Relative paths**: Resolved from the current working directory
- **Absolute paths**: Used as specified

#### Loading Modes

| Mode | Description |
|------|-------------|
| `HIPDNN_PLUGIN_LOADING_ADDITIVE` | Adds new paths to the existing plugin search paths |
| `HIPDNN_PLUGIN_LOADING_ABSOLUTE` | Only loads from the specified paths |

#### Example Usage

```c
// Add custom plugin directories
const char* custom_paths[] = {
    "/home/user/my_plugins",        // Absolute path
    "./local_plugins",              // Relative to working directory
    "/opt/custom/hipdnn/plugins"
};

hipdnnSetEnginePluginPaths_ext(
    3,                              // Number of paths
    custom_paths,                   // Array of path strings
    HIPDNN_PLUGIN_LOADING_ADDITIVE  // Add to existing paths
);
```

Plugins are loaded according to the selected path schema during hipDNN handle creation. Changing paths after handle creation has no effect until another handle is created.

---

## Error Handling

hipDNN provides functions for retrieving error information:

### Getting Error Strings

```c
// Convert status code to string
const char* error_str = hipdnnGetErrorString(status);

// Get detailed error message for the current thread
char message[256]; // The maximum error message size is 256 characters
hipdnnGetLastErrorString(message, sizeof(message));
```

### Best Practices

1. Check return status codes from all hipDNN API calls
2. Use `hipdnnGetLastErrorString` for detailed error context
3. Enable appropriate logging levels during development and debugging
4. Configure logging to files for production deployments
