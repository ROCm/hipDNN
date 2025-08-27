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
- [14. Future Improvements](#14-future-improvements)

## 1. Naming Summary

| Kind | Format | Example |
|------|--------|---------|
| Class / Struct | PascalCase | TensorDescriptor |
| Interface | I + PascalCase | ITensorView |
| Function | camelCase | buildGraph() |
| Variable (local / parameter) | camelCase | workspaceSize |
| Member variable | _camelCase | _cachedPlan |
| Static class variable | s_camelCase | s_engineCount |
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
- Overloads should remain behaviorally symmetrical; prefer explicit helper names instead of ambiguous overload sets when argument meaning changes.

## 4. Variables

- Minimize scope; declare as near first use.
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
- Provide a clear initialization and teardown story if non-trivial.

## 7. Interfaces

- Naming: `IInterfaceName`.
- Keep pure abstract; avoid data members.
- Prefer minimal surface area; segregate roles into multiple interfaces if needed.

## 8. Enums

- Enum type name: PascalCase (e.g. `EngineMode`, `ConvolutionMode`).
- Enumerator names: UPPER_SNAKE (`ENGINE_MODE_DEFAULT`, `ENGINE_MODE_DETERMINISTIC`).
- Keep tokens concise; avoid redundant suffixes unless needed for disambiguation.
- When mirroring external APIs, keep exact enumerator spellings.
- Do not mix styles (no camelCase or PascalCase enumerators).

## 9. Constants

- UPPER_CASE with optional single underscores: `DEFAULT_ALIGNMENT`, `MAX_TENSOR_RANK`.
- Prefer `constexpr` over macros when possible.

## 10. Test Naming Guidelines

GoogleTest reserves underscores in test suite and test names for future expansion. Current repository names with underscores risk future incompatibility; we proactively constrain test suite naming.

Rules below apply ONLY to the TestSuite name (first parameter of `TEST` / `TEST_F`). The TestCase (second parameter) can be descriptive but should still avoid the reserved keywords where noted.

**Ordering & Composition (left → right):**

1. Optional `Integration` prefix for integration tests.
2. Optional `Gpu` (immediately after `Integration` if both apply) for GPU-required tests.
3. Core Feature / Subject under test (PascalCase, no underscores).
4. Optional Shape/Layout token(s) (e.g. `Nhwc`, `Nchw`) BEFORE datatype if datatype is present.
5. Optional Datatype token (`Bfp16`, `Fp16`, `Float`) at the end.

Omit any category that does not apply.

### 10.1 Keywords (reserved positions)

- **Integration** (only for integration tests, always first if present).
- **Gpu** (always first unless preceded by Integration).
- **Datatypes**: Bfp16, Fp16, Float.
- **Layout / Shape** (examples): Nchw, Nhwc (optional).

### 10.2 Valid Examples

```cpp
IntegrationGpuConvolutionPlannerNchwFloat
GpuActivationKernelNchwFloat
GpuExecutionPlanBuilderFloat
GpuExecutionPlanBuilderNchw
IntegrationGraphFusion
ConvolutionHeuristicsFloat
ConvolutionHeuristics
```

### 10.3 Invalid Examples (and why)

| Name | Issue |
|------|-------|
| GpuIntegrationConvolution | Wrong order; Integration must precede Gpu |
| ConvolutionFloatNchw | Layout must precede datatype |
| IntegrationConvolutionGpuFloat | Gpu must directly follow Integration |
| Gpu_Convolution | Underscore not allowed |
| GpuConvolutionFP16 | Datatype token must match exact casing `Fp16` |

### 10.4 Test Case (second parameter)

May be richly descriptive: `HandlesLargeStride`, `RejectsMismatchedLayouts`. Avoid duplicating suite-level keywords (`Integration`, `Gpu`, datatype tokens) redundantly inside the test case name unless clarity requires.

### 10.5 Rationale

- Ordering enforces quick visual parsing (environment → scope → subject → specialization).
- Avoid underscores to remain future-proof with gtest evolution.
- Suffix datatype to emphasize functional context before precision variant.
- Consistent pattern simplifies filtering (e.g. `--gtest_filter=Gpu*Float`).

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
TEST(GpuConvolutionPlannerNchwFloat, HandlesLargeKernels) {
    // ...
}

TEST(IntegrationGpuGraphFusionFloat, FusesThreeSequentialOps) {
    // ...
}

TEST(ConvolutionHeuristics, ChoosesDeterministicPath) {
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

## 14. Future Improvements

- Automated lint / CI rule for test suite naming.
- Scripted audit to migrate legacy underscore-heavy identifiers progressively.

---

Adhering to these rules maintains readability, consistency, and tooling friendliness across the codebase.
