# hipDNN

# EARLY ADOPTER WARNING
**Hipdnn is in the early stages of development, there is currently very limitied or no functionality available to solve problems**

## Getting Started
The fastest way to get started with hipdnn is to follow the [quick start steps in the build guide](./docs/Building.md#quickstart-building-and-installing-hipdnn)

## Building
The full build steps are documented in the [Building.md](./docs/Building.md) file.

## Development Environment
Most of the development team is using vscode.  A few of our setup tips are in the [vscode.md](./docs/vscode.md) file.

## Design
The overall design of hipDNN is documented in the [Design.md](./docs/Design.md) file.  This document includes the overall
architecture of the library.

## Plugin Development
If you are interested in writing a plugin for hipDNN, please see the [Plugin Development](./docs/PluginDevelopment.md) document.

## Logging
There are two environment variables that control the logging in hipDNN. 

`HIPDNN_LOG_LEVEL`: This controls the level of logging.  The supported levels are:
- `off`: no logging
- `info`: info level logging
- `warn`: warning level logging
- `error`: error level logging

If the log level is not set, then the default log level is off.

`HIPDNN_LOG_DIR`: this controls the directory where log files will be written.  If this is not set, log files will be written to stdout.  The backend log file will have the name `hipdnn_timestamp.log` when written to the directory.

### Frontend and Plugin Logging.
Currently the frontend and plugin can use the same logging system as hipdnn backend but they will have to setup the logger themselves.  They will also currently have their own individual log file as there is no callback mechanism in place yet to send the logs to be written by the backend logger.