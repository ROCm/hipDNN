# hipDNN

## ⚠️ EARLY ADOPTER WARNING
**hipDNN is in the early stages of development. There is currently very limited or no functionality available to solve problems.**

---

## Table of Contents

- [Getting Started](#getting-started)
- [Building](#building)
- [Design](#design)
- [Plugin Development](#plugin-development)
- [Logging](#logging)
  - [Frontend and Plugin Logging](#frontend-and-plugin-logging)
- [Project Components](#project-components)
  - [Backend](./backend/README.md)
  - [Frontend](./frontend/README.md)
  - [SDK](./sdk/README.md)
  - [Samples](./samples/README.md)
  - [Tests](./tests/README.md)
  - [Plugins](./plugins/miopen_legacy_plugin/README.md)
- [Documentation](#documentation)
  - [Building Guide](./docs/Building.md)
  - [Design Overview](./docs/Design.md)
  - [Plugin Development Guide](./docs/PluginDevelopment.md)
- [Docker Support](./dockerfiles/README.md)

---

## Getting Started

The fastest way to get started with hipDNN is to follow the [quick start steps in the build guide](./docs/Building.md#quickstart-building-and-installing-hipdnn).

## Building

The full build steps are documented in the [Building.md](./docs/Building.md) file.

## Design

The overall design of hipDNN is documented in the [Design.md](./docs/Design.md) file. This document includes the overall architecture of the library.

## Plugin Development

If you are interested in writing a plugin for hipDNN, please see the [Plugin Development](./docs/PluginDevelopment.md) document.

---

## Project Components

hipDNN is organized into several key components:

- **[Backend](./backend/README.md)**: The core shared library providing a C API for operation graphs
- **[Frontend](./frontend/README.md)**: A header-only C++ API wrapper around the backend
- **[SDK](./sdk/README.md)**: Header-only library for plugin development
- **[Samples](./samples/README.md)**: Example implementations showing how to use hipDNN
- **[Tests](./tests/README.md)**: Integration tests for the public API
- **[Plugins](./plugins/)**: Extensions to hipDNN, including the [MIOpen Legacy Plugin](./plugins/miopen_legacy_plugin/README.md)
