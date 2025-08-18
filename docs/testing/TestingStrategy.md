# hipDNN Testing Strategy

## Overview

This document outlines the comprehensive testing strategy for hipDNN, covering white box testing (unit tests), black box testing (API tests), integration testing, and performance/benchmarking.

---

## 1. White Box Testing (Unit Tests)

### Backend Component Unit Tests

**Location**: `backend/tests/`

**Purpose**: Test internal implementation details of hipDNN backend

**Test Categories**:
- Descriptors
- Plugin system
- Error handling
- Utilities
- Handle
- Graph extensions

**Requirements**:
- Use GMOCK for mocking dependencies
- Use stubbed plugin implementations for plugin testing
- Fast execution
- No GPU testing, or very minimal GPU testing for APIs that require device handles
- GPU operations must be marked with `SKIP_IF_NO_DEVICE()`

**Applicable testing environments**:
- Windows & supported Linux distros
- GPU hardware shouldn't impact these tests

**Frequency of tests**: Run on each PR

### Frontend Component Unit Tests

**Location**: `frontend/tests/`

**Purpose**: Test internal implementation details of hipDNN frontend

**Test Categories**:
- Attribute
- Node
- Graph construction & flow
- Utilities

**Requirements**:
- Use mocked backend for isolation
- Use GMOCK for mocking dependencies
- Fast execution
- No GPU testing, or very minimal GPU testing for APIs that require device handles (hipStreams etc.)
- GPU operations must be marked with `SKIP_IF_NO_DEVICE()`

**Applicable testing environments**:
- Windows & supported Linux distros
- GPU hardware shouldn't impact these tests

**Frequency of tests**: Run on each PR

### SDK Component Unit Tests

**Location**: `sdk/tests/`

**Purpose**: Test internal implementation details of hipDNN SDK

**Test Categories**:
- Plugins
- Data objects
- Logging
- Utilities

**Requirements**:
- Use GMOCK for mocking dependencies
- Fast execution
- No GPU testing, or very minimal GPU testing is expected for the SDK
- Note: This may change in the future if/when GPU reference implementations are added

**Applicable testing environments**:
- Windows & supported Linux distros
- GPU hardware shouldn't impact these tests

**Frequency of tests**: Run on each PR

### Plugin Unit Tests

**Location**: Each plugin's directory (e.g., `plugins/miopen_legacy_plugin/tests/`)

**Purpose**: Test internal implementation details of plugin

**Test Categories**: TBD based on plugin implementation

**Requirements**:
- Use GMOCK for mocking dependencies
- Fast execution
- Minimal & fast GPU testing
- GPU operations should be skippable if machine is CPU only
- Ideally uses small shapes, or reference golden data to do a quick validation of graph executions

**Applicable testing environments**:
- Windows & supported Linux distros
- Test on all ASICs supported by the plugin

**Frequency of tests**: Run on each PR

---

## 2. Black Box Testing (API Tests)

### Backend API Tests

**Location**: `tests/backend/`

**Purpose**: Validate API of hipDNN backend works as expected

**Test Categories**:
- Descriptor create, get/set properties, and destroy for:
  - Engine API
  - Engine config API
  - Engine heuristic API
  - Execution plan API
  - Handle API
  - Variant pack API
  - Graph API
  - Graph extension API for serialized graph structures
- Backend execute API
- Plugin management extension API

**Requirements**:
- Test only public interfaces from `backend/include/`
- Use stubbed plugins for controlled testing
- Fast running
- No GPU testing, or very minimal GPU testing for APIs that require device handles
- GPU operations must be marked with `SKIP_IF_NO_DEVICE()`

**Applicable testing environments**:
- Windows & supported Linux distros
- GPU hardware shouldn't impact these tests

**Frequency of tests**: Run on each PR

---

## 3. Integration Testing

### Frontend-Backend Integration

**Location**: `tests/frontend/`

**Purpose**: Validate end-to-end hipDNN works as expected

**Test Categories**:
- Graph creation & execution API
- Backend descriptor creation from frontend
- Execution flow validation

**Requirements**:
- Use fake plugins for controlled behavior
- No accuracy or solution validation (stubbed behavior)
- Fast running
- No GPU testing, or very minimal GPU testing for APIs that require device handles
- GPU operations must be marked with `SKIP_IF_NO_DEVICE()`

**Applicable testing environments**:
- Windows & supported Linux distros
- GPU hardware shouldn't impact these tests

**Frequency of tests**: Run on each PR

### Plugin Integration Tests

**Location**: Each plugin's directory (e.g., `plugins/miopen_legacy_plugin/integration_tests/`)

**Purpose**: Validate end-to-end graph support for plugin implementation

**Test Categories**: TBD based on plugin implementation

**Requirements**:
- Can be slower running
- Will require GPU
- Should validate correctness and graph support
- Each plugin maintains its own test suite

**Applicable testing environments**:
- Windows & supported Linux distros
- Test on all ASICs supported by the plugin

**Frequency of tests**: Run on each PR

---

## 4. Performance/Benchmarking

**Location**: Separate project for benchmarking full hipDNN install (To be determined once made)

**Purpose**:
- Track performance of hipDNN & installed plugins across a broad set of graphs
- Track accuracy of hipDNN & installed plugins across a broad set of graphs

> **Note**: Each plugin will have integration tests for functionality that it supports, and this suite will be the full integration set of shapes that runs across plugins.

**Test categories**:
- A quick running set of graphs to run per PRs to flag severe regressions
- A long running set of graphs to run on demand to flag broad regressions

**Requirements**:
- Minimal set of graphs are maintained to be used as pre-checkin performance check
- Full set of graphs are maintained to be used for on-demand performance & accuracy checks
- Requires GPU
- Validates correctness and performance of graphs

**Applicable testing environments**:
- Windows & supported Linux distros
- Test on all ASICs supported by hipDNN
- Note: Certain plugins/graphs may have ASIC restrictions

**Frequency of tests**:
- Minimal graph suite will run on each PR to catch obvious regressions
- Full graph suite will be runnable on demand, and run nightly or weekly to catch regressions (running frequency TBD)

---

## General Testing Requirements

### Code Coverage
- hipDNN has a code coverage target of **80% overall**
- Each sub-section should be above 80% individually
- The overall target needs to remain above 80%

### Test Environment Compatibility
Tests need to work in the following environments:
- CLI in a build environment via `make check`, `ninja check`, `make check_ctest`, `ninja check_ctest`
- Visual Studio Code and extensions like TestMate
- Installed testing artifacts
- Running the built test executables

### GPU Requirements
- Without a GPU: All tests requiring a GPU should be skippable if a GPU is not present, and not cause errors to be displayed (warnings instead)
- Windows & supported Linux distros

---

## Current Testing Status

- Frontend, SDK, backend, and MIOpen plugin all have full Whitebox, Blackbox, and integration testing with code coverage above 80% target
- Currently missing golden data tests to verify reference implementations
- Benchmarking & performance testing project does not exist yet

---

## Future Improvement Roadmap

1. Create standardized naming conventions for tests
2. Document best practices, patterns, and requirements for new tests
3. Add ASAN as an automatic step to CI
4. Add golden reference data to use for unit testing at plugin level & to verify reference implementations
5. Swap to leverage TheRock for CI
6. Add installable testing artifacts
7. Create a benchmarking and performance project for capturing performance and accuracy for full hipDNN graphs
