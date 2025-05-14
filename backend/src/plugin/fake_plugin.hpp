// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
#include "descriptors/variant_descriptor.hpp"
#include "engine_ids.hpp"
#include "handle/handle.hpp"
#include "hipdnn_plugin_base.hpp"

namespace hipdnn_backend
{
// This is a placeholder implementation for the plugin.
// It allows integration with higher-level API handlers while the actual plugin is under development.
// This implementation serves as a functional mock to facilitate setup and connection of components,
// but it is not the final or fully functional version. Its mock version.
struct Fake_plugin : Hipdnn_plugin_base
{
    ~Fake_plugin() override = default;

    std::set<int64_t> get_applicable_engines(Graph_descriptor* graphdesc) override
    {
        (void)graphdesc;

        return {HIPDNN_ENGINE_ID_FAKE};
    }

    void execute(Graph_descriptor* graphdesc,
                 Variant_descriptor* vpack,
                 hipdnnHandle* handle) override
    {
        // calls gpu kernel to execute the graph using C API?
        (void)graphdesc;
        (void)vpack;
        (void)handle;
    }

    int64_t get_max_workspace_size(Graph_descriptor* graphdesc, int64_t engine_id) override
    {
        (void)graphdesc;
        (void)engine_id;
        return 1024;
    }
};

}