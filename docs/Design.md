# hipDNN Design

Note: We are very early on in the development of hipDNN.  This design may change as we learn about new requirements.

hipDNN is a multi-operation fusion engine, these fusions are to provide improved performance.  hipDNN uses operation graphs, the graphs are essentially a language that is used to describe a kernel.  hipDNN will have multiple different backend engines, heuristics will be used to select the fastest implemenation for a given graph.  hipDNN will support filtering in order to control which engine is selected.  hipDNN has an API that matches the industry standard for deep learning libraries.

## High-Level Architecture 

hipDNN has a plugin-based architecture in order to provide many different solvers. This enables users to extend hipDNN to support new routines/algorithms without modifying the core library.

![hipDNN Architecture](./images/hipDNN_Architecture.png)

### Users
**External Users**: Users who will leverage the Frontend API.

**Advanced Users**: Users who will use both the frontend and backend APIs depending on the needs of their application.

### Components
**Frontend**: A header only library that provides the industry standard API for interacting with hipDNN.  The frontend is mainly a wrapper around the backend api to provide a C++ API for integration.

**Backend**: A shared library which provides a C API for hipDNN.  The backend is the main component of hipDNN which connects problems to plugins that can solve them. In the future the C-API will eventually have both an imperative API and a graph API.

**Plugin SDK**: A headery-only library that plugins will integrate against in order to be used by hipDNN.

**MIOpen Legacy Plugin**: A plugin that wraps MIOpen and provides solvers for Convolution and BatchNorm graphs. This is meant to support all existing MIOpen capabilities.

**Other Plugins**: Eventually there will be other plugins available that provide different solvers than MIOpen.  Each additional plugin will enable hipDNN to solve a greater set of problems or solve current problems faster.
