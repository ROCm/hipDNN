# hipDNN Testing Strategy

This document outlines the comprehensive testing strategy for hipDNN, covering white box testing (unit tests), black box testing (API tests), integration testing, and performance/benchmarking.

---

## 1. White Box Testing (Unit Tests)

White box tests focus on internal implementation details of hipDNN components. All white box tests should run on each PR.

### Component Comparison

| Component | Location | Purpose | GPU Testing | Environments |
|-----------|----------|---------|-------------|--------------|
| **Backend** | `backend/tests/` | Test internal implementation of hipDNN backend | Minimal/None - mark with `SKIP_IF_NO_DEVICE()` | Windows & Linux |
| **Frontend** | `frontend/tests/` | Test internal implementation of hipDNN frontend | Minimal/None - mark with `SKIP_IF_NO_DEVICE()` | Windows & Linux |
| **SDK** | `sdk/tests/` | Test internal implementation of hipDNN SDK | Minimal/None expected | Windows & Linux |
| **Plugin** | `plugins/<name>/tests/` | Test internal implementation of specific plugin | Minimal & fast - skippable if CPU only | Windows & Linux |

---

### Test Categories by Component

#### Backend
- Descriptors
- Plugin system
- Error handling
- Utilities
- Handle
- Graph extensions

#### Frontend
- Attribute
- Node
- Graph construction & flow
- Utilities

#### SDK
- Plugins
- Data objects
- Logging
- Utilities

#### Plugin
- TBD based on plugin implementation

### Common Requirements

- **Mocking**: Use GMOCK for mocking dependencies
- **Execution**: Fast execution required
- **Isolation**: Use stubbed/mocked implementations for dependencies
- **GPU Operations**: Must be marked with `SKIP_IF_NO_DEVICE()`
- **Coverage**: Each component should maintain >80% code coverage

---

## 2. Black Box Testing (API Tests)

Black box tests validate the public API without knowledge of internal implementation.

### Backend API Tests

| Attribute | Details |
|-----------|---------|
| **Location** | `tests/backend/` |
| **Purpose** | Validate API of hipDNN backend works as expected |
| **Requirements** | • Test only public interfaces from `backend/include/`<br>• Use stubbed plugins for controlled testing<br>• Fast running<br>• GPU operations marked with `SKIP_IF_NO_DEVICE()` |
| **Environments** | Windows & supported Linux distros |
| **Frequency** | Run on each PR |

#### Test Categories
- Descriptor APIs (create, get/set properties, destroy)
  <!-- - Engine API
  - Engine config API
  - Engine heuristic API
  - Execution plan API
  - Handle API
  - Variant pack API
  - Graph API
  - Graph extension API for serialized graph structures -->
- Backend execute API
- Plugin management extension API

---

## 3. Integration Testing

Integration tests validate end-to-end functionality across components.

### Integration Test Comparison

| Test Type | Location | Purpose | GPU Required | Test Speed |
|-----------|----------|---------|--------------|------------|
| **Frontend-Backend** | `tests/frontend/` | Validate end-to-end hipDNN functionality | No - mark GPU ops with `SKIP_IF_NO_DEVICE()` | Fast |
| **Plugin Integration** | `plugins/<name>/integration_tests/` | Validate end-to-end graph support for plugin | Yes - required for validation | Can be slower |

### Test Requirements by Type

| Test Type | Key Requirements |
|-----------|-----------------|
| **Frontend-Backend** | • Use fake plugins for controlled behavior<br>• No accuracy/solution validation (stubbed)<br>• Test graph creation & execution API<br>• Test backend descriptor creation from frontend<br>• Test execution flow validation |
| **Plugin Integration** | • Validate correctness and graph support<br>• Each plugin maintains its own test suite<br>• Test on all ASICs supported by the plugin<br>• Can include performance validation |

---

## General Testing Requirements

### Code Coverage
- **Target**: 80% overall coverage
- **Component Target**: Each sub-section should be above 80% individually
- **Enforcement**: Coverage must remain above 80% for PRs to be accepted

### Test Environment Compatibility

Tests must work in the following environments:

| Environment Type | Supported Methods |
|-----------------|-------------------|
| **CLI Build Environment** | `make check`, `ninja check`, `make check_ctest`, `ninja check_ctest` |
| **IDE** | Visual Studio Code and extensions like TestMate |
| **Artifacts** | • Installed testing artifacts<br>• Running built test executables |

### GPU Requirements
- **Without GPU**: All GPU tests must be skippable (warnings, not errors)
- **With GPU**: Tests should detect and utilize available GPU resources
- **Platform Support**: Windows & supported Linux distributions

---

<!-- 
## 4. Performance/Benchmarking (Future Work)

**Status**: Not yet implemented - planned for future development

**Location**: Separate project for benchmarking full hipDNN install (TBD)

**Purpose**:
- Track performance of hipDNN & installed plugins across a broad set of graphs
- Track accuracy of hipDNN & installed plugins across a broad set of graphs

**Test categories**:
- Quick running set of graphs per PR (severe regression detection)
- Long running set of graphs on demand (broad regression detection)

**Frequency**:
- Minimal suite: Per PR
- Full suite: On demand, nightly or weekly (TBD)
-->

<!--
## Current Testing Status

- Frontend, SDK, backend, and MIOpen plugin all have full Whitebox, Blackbox, and integration testing with code coverage above 80% target
- Currently missing golden data tests to verify reference implementations
- Benchmarking & performance testing project does not exist yet

## Future Improvement Roadmap

1. Create standardized naming conventions for tests
2. Document best practices, patterns, and requirements for new tests
3. Add ASAN as an automatic step to CI
4. Add golden reference data to use for unit testing at plugin level & to verify reference implementations
5. Swap to leverage TheRock for CI
6. Add installable testing artifacts
7. Create a benchmarking and performance project for capturing performance and accuracy for full hipDNN graphs
-->
