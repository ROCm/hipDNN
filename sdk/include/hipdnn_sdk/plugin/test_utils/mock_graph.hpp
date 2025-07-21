// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>

namespace hipdnn_plugin
{

class Mock_graph : public Graph_interface
{
public:
    MOCK_METHOD(const hipdnn_sdk::data_objects::Graph&, get_graph, (), (const, override));
    MOCK_METHOD(bool, is_valid, (), (const, override));
    MOCK_METHOD(uint, node_count, (), (const, override));
    MOCK_METHOD(bool,
                has_only_supported_attributes,
                (std::set<hipdnn_sdk::data_objects::NodeAttributes> supported_attributes),
                (const, override));
    MOCK_METHOD(const hipdnn_sdk::data_objects::Node&, get_node, (uint index), (const, override));
    MOCK_METHOD(
        (const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&),
        get_tensor_map,
        (),
        (const, override));
};

}