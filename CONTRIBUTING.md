# Contributing to hipDNN
## Overview
All changes to the hipDNN codebase will need the following before they will be accepted:
- All code must follow the format as specified by the clang-format file.
- All code must be covered by unit tests.
- If applicable, integration tests should be added to verify the full solution.
- Code must be warning free and free of any clang-tidy errors.
- Relevent documentation must be updated to reflect the changes.

### Example of adding or extending functionality
- Add new schema for the attributes that represent the operation being added to hipDNN into sdk/schemas
- Update the NodeAttributes union inside graph.fbs to have the new attributes.
- Add new node and attributes to hipDNN frontend
- Add a method to the frontend graph for adding the new node.
- Add a new plugin, or extend an existing plugin to implement handling the newly added operation.
    - Note:
        - Some plugins will dynamically generate kernels, and will be able to handle patterns
            e.g. PointwiseOp* -> conv -> PointwiseOp*
        - Some plugins will have high performance static kernels, and will be handling only specific graphs
            e.g. Single ops like conv, batchnorm, etc.
- Add a new sample to the samples folder to cover the new functionality.

**Dont forget to ensure you have met all base repository requirements before submitting a pull request. See [Overview](#overview)**

### Adding/Modifying a Plugin
See [Plugin Development](./docs/PluginDevelopment.md) for more information on how to add a new plugin or modify an existing one.


