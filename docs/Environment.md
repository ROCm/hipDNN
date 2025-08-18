## Logging

There are two environment variables that control the logging in hipDNN:

### `HIPDNN_LOG_LEVEL`

This controls the level of logging. The supported levels are:
- `off`: no logging
- `info`: info level logging
- `warn`: warning level logging
- `error`: error level logging
- `fatal`: fatal level logging

If the log level is not set, then the default log level is `off`.

### `HIPDNN_LOG_FILE`

This controls the path to the file where logs will be **appended**. If this is not set, logs will go to `stderr`.

### Frontend and Plugin Logging

The frontend and plugins log to the same file as the backend but they will have to setup the logger themselves via the `initialize_callback_logging` function. The user should pass the backend API call `hipdnnLoggingCallback_ext` as the callback function to have the logs all output to the same destination.