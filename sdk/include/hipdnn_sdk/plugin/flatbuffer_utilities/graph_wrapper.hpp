// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>

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
        get_tensor_map() const
        = 0;
};

class Graph_wrapper : public Graph_interface
{
public:
    explicit Graph_wrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::Graph>())
            {
                _shallow_graph = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::Graph& get_graph() const override
    {
        throw_if_not_valid();
        return *_shallow_graph;
    }

    bool is_valid() const override
    {
        return _shallow_graph != nullptr;
    }

    uint node_count() const override
    {
        throw_if_not_valid();
        return _shallow_graph->nodes()->size();
    }

    bool has_only_supported_attributes(
        std::set<hipdnn_sdk::data_objects::NodeAttributes> supported_attributes) const override
    {
        throw_if_not_valid();

        // NOLINTNEXTLINE(readability-use-anyofallof)
        for(const auto node : *_shallow_graph->nodes())
        {
            if(!supported_attributes.contains(node->attributes_type()))
            {
                return false;
            }
        }
        return true;
    }

    const hipdnn_sdk::data_objects::Node& get_node(uint index) const override
    {
        throw_if_not_valid();

        if(index >= _shallow_graph->nodes()->size())
        {
            throw std::out_of_range("Index out of range for graph nodes");
        }

        return *_shallow_graph->nodes()->Get(index);
    }

    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        get_tensor_map() const override
    {
        throw_if_not_valid();

        if(!_tensor_map.empty())
        {
            return _tensor_map;
        }

        for(const auto tensor : *_shallow_graph->tensors())
        {
            _tensor_map[tensor->uid()] = tensor;
        }

        return _tensor_map;
    }

private:
    void throw_if_not_valid() const
    {
        if(!is_valid())
        {
            throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                         "Graph is not valid");
        }
    }

    // Pointer to the flatbuffer representation of the graph. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_sdk::data_objects::Graph* _shallow_graph = nullptr;

    //lazy init state;
    mutable std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>
        _tensor_map;
};

}