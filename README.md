# hipDNN-internal

## Repo setup todo
- clang format
  - format the entire repo
- front end portion
- front end integration tests
- call integration tests just test, then add readme for whats expected.

## VSCode Setup

### CMake
- Install the cmake extension: 
- Tell it to scan recursively for compilers and give it the /opt/rocm/llvm directory.
- Once it has been found you can select CLANG instead of GCC.

### Test Explorer
You can use https://github.com/matepek/vscode-catch2-test-adapter to have vscode test explorer auto load the
gtest binaries into the editor.