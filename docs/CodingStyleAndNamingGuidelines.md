# Coding Style & Naming Guidelines

This document defines the canonical project-wide coding and test naming conventions.

## Table of Contents

- [1. Naming Summary](#1-naming-summary)
- [2. File & Class Naming](#2-file--class-naming)
- [3. Functions](#3-functions)
- [4. Variables](#4-variables)
- [5. Members](#5-members)
- [6. Globals](#6-globals)
- [7. Interfaces](#7-interfaces)
- [8. Enums](#8-enums)
- [9. Constants](#9-constants)
- [10. Test Naming Guidelines](#10-test-naming-guidelines)
  - [10.1 Keywords (reserved positions)](#101-keywords-reserved-positions)
  - [10.2 Valid Examples](#102-valid-examples)
  - [10.3 Invalid Examples (and why)](#103-invalid-examples-and-why)
  - [10.4 Test Case (second parameter)](#104-test-case-second-parameter)
  - [10.5 Rationale](#105-rationale)
- [11. Examples](#11-examples)
  - [Class & File](#class--file)
  - [Interface](#interface)
  - [Constant & Enum](#constant--enum)
  - [Test (gtest)](#test-gtest)
- [12. Decision Checklist](#12-decision-checklist)
- [13. Deviation Process](#13-deviation-process)
- [14. Automated Tooling](#14-automated-tooling)
  - [14.1 Clang-Tidy Rules](#141-clang-tidy-rules)
  - [14.2 Test Naming Enforcement Tool](#142-test-naming-enforcement-tool)

## 1. Naming Summary

| Kind | Format | Example |
|------|--------|---------|
| Class / Struct | PascalCase | TensorDescriptor |
| Interface | I + PascalCase | ITensorView |
| Function | camelCase | buildGraph() |
| Variable (local / parameter) | camelCase | workspaceSize |
| Private Member variable | _camelCase | _cachedPlan |
| Static variable | s_camelCase | s_engineCount |
| Global variable | g_camel_case | g_global_state |
| Constant / Macro | UPPER_CASE | MAX_WORKSPACE_BYTES |
| Enum Type | PascalCase | EngineMode |
| Enum Value | UPPER_SNAKE | ENGINE_MODE_DEFAULT |

## 2. File & Class Naming

- Prefer the filename to match the primary class it contains: `ExecutionPlan.hpp`, `ExecutionPlan.cpp`.
- Utility collections (no single dominant class) may use a name that makes sense to describe the contents (e.g. `Error.hpp`).
- Interfaces: prefix `I`, e.g. `IAllocator.hpp`.
- Keep one major class per file when practical.
- Avoid excessively long filenames; lean on directory structure for grouping.

## 3. Functions

- Use descriptive action-oriented verbs: `createPlan`, `finalizeConfig`, `launchKernels`.

## 4. Variables

- Favor clarity over abbreviation: prefer `intermediateSize` to `intSz`.

## 5. Members

- Private / protected data members: prefix single underscore `_` then camelCase (`_opGraph`).
- Static data members: `s_camelCase`.
- Exposed constants inside a class: `static constexpr` UPPER_CASE.
- Plain structs whose intent is a passive aggregate (all or mostly public data) DO NOT prefix member names with `_`; just use camelCase.
  - Rationale: underscores communicate encapsulation; POD-style structs are transparent.

Example:

```cpp
struct TensorExtent {
    int n;
    int c;
    int h;
    int w;
};
```

If later you add invariants or non-trivial behavior, consider converting to a class and applying the underscore rule to newly private members.

## 6. Globals

- Avoid unless absolutely required; prefix `g_` to make visibility explicit.

## 7. Interfaces

- Naming: `IInterfaceName`.
- Keep pure abstract; avoid data members.

## 8. Enums

- Enum type name: PascalCase (e.g. `EngineMode`, `ConvolutionMode`).
- Enumerator names: UPPER_SNAKE (`ENGINE_MODE_DEFAULT`, `ENGINE_MODE_DETERMINISTIC`).
- When mirroring external APIs, keep exact enumerator spellings.

## 9. Constants

- UPPER_CASE with optional single underscores: `DEFAULT_ALIGNMENT`, `MAX_TENSOR_RANK`.
- Prefer `constexpr` over macros when possible.

## 10. Test Naming Guidelines

GoogleTest reserves underscores in test suite and test names for future expansion. Current repository names with underscores risk future incompatibility; we proactively constrain test suite naming.

Rules below apply ONLY to the TestSuite name (first parameter of `TEST` / `TEST_F` / `TEST_P`). The TestCase (second parameter) can be descriptive but should still avoid the reserved keywords where noted.

**Enforced Ordering & Composition (left → right):**

1. Optional `Integration` prefix for integration tests.
2. Optional `Gpu` (immediately after `Integration` if both apply) for GPU-required tests.
3. Core Feature / Subject under test (PascalCase, no underscores).
4. Optional Datatype token (`Bfp16`, `Fp16`, `Fp32`) at the end.

Omit any category that does not apply.

### 10.1 Keywords (reserved positions)

- **Integration** (only for integration tests, always first if present).
- **Gpu** (always first unless preceded by Integration).
- **Datatypes**: Bfp16, Fp16, Fp32.

### 10.2 Unit Tests

In most cases unit style tests should be named so the directly mirror the class under test.  If the class is named `MyClass`, then the test suite should be named `TestMyClass`.  In general these kinds of tests should try to avoid using anything that requires Gpu support.  This is not always possible, in the cases where Gpu support is required, the test suite should be named `GpuTestMyClass`.

### 10.3 Valid Examples

```cpp
IntegrationGpuConvolutionPlannerNchwFp32
GpuTestActivationKernelNchwFp32
GpuTestExecutionPlanBuilderFp32
GpuTestExecutionPlanBuilderNchw
IntegrationGraphFusion
TestConvolutionHeuristicsFp32
TestConvolutionHeuristics
```

### 10.4 Invalid Examples (and why)

| Name | Issue |
|------|-------|
| GpuIntegrationConvolution | Wrong order; Integration must precede Gpu |
| ConvolutionFp32Nchw | Layout must precede datatype |
| IntegrationConvolutionGpuFp32 | Gpu must directly follow Integration |
| Gpu_Convolution | Underscore not allowed |
| GpuConvolutionFP16 | Datatype token must match exact casing `Fp16` |

### 10.5 Test Case (second parameter)

May be richly descriptive: `HandlesLargeStride`, `RejectsMismatchedLayouts`. Avoid duplicating suite-level keywords (`Integration`, `Gpu`, datatype tokens) redundantly inside the test case name unless clarity requires.  The test case name is the preferred place to list the shape/layout variant being tested (e.g. `Nchw`, `Nhwc`).

### 10.6 Test File Naming

The test file name should mirror the primary test suite it contains.  For example, if the main test in a suite is `TestMyClass`, the file should be named `TestMyClass.cpp`.  That same file may also contain `GpuTestMyClass` but it is not the primary test suite so the file name does not need to reflect it.  For integration tests, the main test suite might be named `IntegrationGpuFeatureX` but have several child suites like `IntegrationGpuFeatureXFp32` and `IntegrationGpuFeatureXBfp16`. The parent suite name is the primary suite, so the file name should be `IntegrationGpuFeatureX.cpp`. 

### 10.7 Rationale

- Ordering enforces quick visual parsing (environment → scope → subject → specialization).
- Avoid underscores to remain future-proof with gtest evolution.
- Suffix datatype to emphasize functional context before precision variant.
- Consistent pattern simplifies filtering (e.g. `--gtest_filter=Gpu*Fp32`).

## 11. Examples

### Class & File

File: `ExecutionPlan.hpp`
```cpp
class ExecutionPlan {
public:
    static constexpr int MAX_STEPS = 8;

    explicit ExecutionPlan(int initialSteps);
    void buildGraph();
    int stepCount() const;

private:
    int _stepCount;
    bool _isFinalized;
};
```

### Interface

```cpp
class IAllocator {
public:
    virtual ~IAllocator() = default;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
};
```

### Constant & Enum

```cpp
enum EngineMode {
    ENGINE_MODE_DEFAULT = 0,
    ENGINE_MODE_DETERMINISTIC = 1
};

constexpr size_t MAX_WORKSPACE_BYTES = 1ull << 32;
```

### Test (gtest)

```cpp
TEST(IntegrationGpuGraphFusionFp32, FusesThreeSequentialOps) {
    // ...
}
TEST(IntegrationGpuGraphFusionBfp16, FusesThreeSequentialOps) {
    // ...
}

TEST(TestExecutionPlan, BuildsGraphCorrectly) {
    // ...
}
```

## 12. Decision Checklist

When adding new code, verify:
- Names follow the table in Section 1.
- File name matches main class (or is a justified utility collection).
- Test suite names follow ordering & allowed tokens.
- Layout tokens appear before datatype tokens when both used.
- No stray underscores in test suite names.

## 13. Deviation Process

If an external API or standard library interop forces divergence (e.g., fixed enum value names), document the exception with a brief comment near the declaration.

## 14. Automated Tooling

The repository includes automated tooling to enforce coding standards and maintain consistency across the codebase.

### 14.1 Clang-Tidy Rules

The project uses clang-tidy to automatically enforce many of the coding style guidelines defined in this document. The configuration can be found in `.clang-tidy` at the repository root.

The CI pipeline automatically runs clang-tidy on all pull requests to ensure compliance before merging.

### 14.2 Test Naming Enforcement Tool

*[Placeholder: A dedicated test naming enforcement tool is planned to automatically validate that all test names follow the conventions outlined in Section 10]*

This tool will:
- Parse all test files and extract TEST/TEST_F suite names
- Validate ordering of keywords (Integration, Gpu, Feature, Layout, Datatype)
- Check for prohibited underscores in test suite names
- Generate reports on non-compliant test names
- Integrate with CI to block merges with invalid test names

---

Adhering to these rules and utilizing the automated tooling maintains readability, consistency, and tooling friendliness across the codebase.
