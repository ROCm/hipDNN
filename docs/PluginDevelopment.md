# Plugin Development

Note: Currently we are very early in the development process of hipDNN.  The first plugin planned is a MIOpen
Legacy Plugin.  This documenation will be updated once we have finalized the plugin API and have a working example.

## hipDNN-SDK library
The hipDNN-SDK API is a Header-Only C++ library which provides the requirments needed in order to create a plugin hipDNN can consume. 

## Plugin API
- Graphs will be passed in a serialized format to the plugins.  Plugins need to use the SDK data objects in order to deserialize the graphs.  From there, the plugin needs to determine if it can solve the graph.  The plugin will then report back to hipDNN if it is able to execute the graph.