When generating code, use Leading_upper_snake_case for class and struct names, use lower_case for functions, variables, and private members.  Prefix private members with _.

Always add copywrite lines "Copyright © Advanced Micro Devices, Inc., or its affiliates." and "SPDX-License-Identifier:  MIT" to the top of all source files.  Ensure you add those whenever you generate a file.

When generating header files, add #pragma once at the top after the copywrite

We use CMake for managing our c/c++ dependencies.  When talking about dependencies, always use cmake.

We use Flatbuffers for serialization, if possible, please respond with flatbuffer relevent responses.

Use gtest when creating C/C++ tests. Dont generate a main function.

Use auto when initializing with a cast to avoid duplicating the type name.