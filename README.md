# hipDNN-internal

## Repo setup todo
- front end portion
- front end integration tests
- Integrate coverage build stuff
- rename rocroller stuff

## Ci Setup todo
- Ask to get coverage build setup
  - Make a warning occur if coverage for new code is < 80%.  Do not fail the build, to meet deadlines, we may have to take some shortcuts
- Ask to have release and debug builds occur
    - Also have tests run with these
- Ask to get clang format scanning as well.

## VSCode Setup

### CMake
- Install the cmake extension: 
- Tell it to scan recursively for compilers and give it the /opt/rocm/llvm directory.
- Once it has been found you can select CLANG instead of GCC.

### Test Explorer
You can use https://github.com/matepek/vscode-catch2-test-adapter to have vscode test explorer auto load the
gtest binaries into the editor