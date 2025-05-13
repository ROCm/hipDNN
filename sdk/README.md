# hipDNN-sdk


## Logging
hipDNN is using spdlog-header-only library to do its logging. The logging utiltiy functions can be used to easily setup a log for your application. 

## Plugin
The plugin folder of the sdk contains the header file plugin_api.h that each plugin must implement. Please see [Plugin Development](../docs/PluginDevelopment.md) for more information on the plugins.

## Schema based data objects
- The hipDNN sdk is using schema based https://flatbuffers.dev/ data objects for describing the graph, and operations.

### How to change schema files
- Adding, or updating the files inside schemas (*.fbs) requires regenerating the output files.
- To regerneate the output files run the make target `make generate_hipdnn_sdk_headers`.
- This will regenarate the files inside include.
- Note: Any changes made to *.fbs files will require regenerating the files, please run the make command anytime the schema files are updated.

## Test Utilities
- Test utilities which are useful for testing hipDNN components. 

## Consuming the hipDNN-SDK using CMake
When hipDNN is installed on a system, this library also gets installed along with .cmake files that can be used.  If you are using CMake, you can find the package using `find_package(hipdnn_sdk)`.  If you cannot find the package after installation, ensure your cmake modules has the install path.  By default on linux systems, hipdnn cmake files are installed to `/opt/rocm/lib/cmake`.