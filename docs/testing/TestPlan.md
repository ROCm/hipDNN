# hipDNN Test Plan

## Table of Contents

- [Prerequisites](#prerequisites)
  - [Test Case 1: CI Is Green](#test-case-1-ci-is-green)
  - [Test Case 2: Documentation is Current](#test-case-2-documentation-is-current)
- [Regular Tests](#regular-tests)
  - [Test Case 1: Run the Automated Tests](#test-case-1-run-the-automated-tests)
- [ASAN Enabled Tests](#asan-enabled-tests)
  - [Test Case 1: Run the Automated Tests with ASAN Enabled](#test-case-1-run-the-automated-tests-with-asan-enabled)

---

## Prerequisites

### Test Case 1: CI Is Green

All existing test plan checks should be running automatically on all PRs against the hipDNN Repo & Develop after merging a PR:
- Pre-checkin does unit & integration tests
- Codecov stage checks code coverage
- Debug runs pre-checkin checks in a debug build

### Test Case 2: Documentation is Current

- Check versions
- Do a light pass on wording etc.
- Changelog is correct

---

## Regular Tests

### Test Case 1: Run the Automated Tests

**Steps**:
1. Clone repo
2. Create build directory:
   ```bash
   mkdir build
   cd build
   ```
3. Configure with CMake:
   ```bash
   cmake ..
   ```
   > **Note**: Add `-DBUILD_ADDRESS_SANITIZER=ON` to cmake command if you want to check with address sanitizer. Running with address sanitizer will disable GPU tests.

4. Run tests:
   ```bash
   ninja check_ctest
   # or
   make check_ctest
   ```

**Expected Results**:
- All tests pass
- Depending on the environment being ran, some GPU tests may be skipped
- For environments without a GPU: all GPU tests should be skipped, but none of them should be failing
- For environments with a GPU: the plugin integration tests will potentially skip if support does not exist for that GPU. However, it's expected that they should provide a skipped message indicating it was skipped due to lacking support for that ASIC

> **Note**: ASIC specific coverage will be determined by the plugin, and is not a global requirement for hipDNN. I.E You can have a plugin that is only expected to support some ASICs.

---

## ASAN Enabled Tests

### Test Case 1: Run the Automated Tests with ASAN Enabled

**Steps**:
1. Clone repo
2. Create build directory:
   ```bash
   mkdir build
   cd build
   ```
3. Configure with ASAN:
   ```bash
   cmake .. -DBUILD_ADDRESS_SANITIZER=ON
   ```
4. Run tests:
   ```bash
   ninja check_ctest
   # or
   make check_ctest
   ```

**Expected Results**:
- All tests pass
- All GPU tests are expected to be skipped since ASAN disables GPU related tests
