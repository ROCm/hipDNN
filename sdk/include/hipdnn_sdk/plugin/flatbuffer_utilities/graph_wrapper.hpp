// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_plugin
{

class Graph_interface
{
public:
    virtual ~Graph_interface() = default;

    virtual const hipdnn_sdk::data_objects::Graph& get_graph() const = 0;
    virtual bool is_valid() const = 0;
    virtual uint node_count() const = 0;
    virtual bool has_only_supported_attributes(
        std::set<hipdnn_sdk::data_objects::NodeAttributes> supported_attributes) const
        = 0;
    virtual const hipdnn_sdk::data_objects::Node& get_node(uint index) const = 0;
    virtual const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        get_tensor_map()
        = 0;
};

class Graph_wrapper : public Graph_interface
{
public:
    explicit Graph_wrapper(const void* buffer, size_t size)
    {
        if(buffer)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::Graph>())
            {
                _graph = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::Graph& get_graph() const override
    {
        return *_graph;
    }

    bool is_valid() const override
    {
        return _graph != nullptr;
    }

    uint node_count() const override
    {
        return _graph ? _graph->nodes()->size() : 0;
    }

    bool has_only_supported_attributes(
        std::set<hipdnn_sdk::data_objects::NodeAttributes> supported_attributes) const override
    {
        if(!_graph)
            return false;

        for(const auto node : *_graph->nodes())
        {
            if(supported_attributes.find(node->attributes_type()) == supported_attributes.end())
                return false;
        }
        return true;
    }

    const hipdnn_sdk::data_objects::Node& get_node(uint index) const override
    {
        if(!_graph || index >= _graph->nodes()->size())
            throw std::out_of_range("Index out of range for graph nodes");

        return *_graph->nodes()->Get(index);
    }

    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        get_tensor_map() override
    {
        if(!_graph)
            throw std::runtime_error("Graph is not valid");

        if(!_tensor_map.empty())
        {
            return _tensor_map;
        }

        for(const auto tensor : *_graph->tensors())
        {
            _tensor_map[tensor->uid()] = tensor;
        }

        return _tensor_map;
    }

private:
    const hipdnn_sdk::data_objects::Graph* _graph = nullptr;

    //lazy init state;
    std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*> _tensor_map;
};

}