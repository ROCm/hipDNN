# Integration Tests for hipDNN

This directory contains integration tests for the **hipDNN** library. The purpose of these tests is to ensure the correctness and reliability of the **publicly exposed API**.

## Guidelines

- Only the **public API** of hipDNN should be tested here
- Avoid testing internal or private implementation details (those should be tested as unit tests)

## Test Structure

- **Backend Tests**: [backend/](./backend/) - Tests for the backend C API
- **Frontend Tests**: [frontend/](./frontend/) - Tests for the frontend C++ API
- **Test Plugins**: [test_plugins/](./test_plugins/) - Mock plugins used for testing

## Testing Documentation

For comprehensive testing information, please refer to:
- **[Testing Strategy](../docs/testing/TestingStrategy.md)** - Overview of hipDNN's testing approach
- **[Test Plan](../docs/testing/TestPlan.md)** - Detailed test cases and procedures
- **[Test Run Template](../docs/testing/TestRunTemplate.md)** - Template for recording test results
