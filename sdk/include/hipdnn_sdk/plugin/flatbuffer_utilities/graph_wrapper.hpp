// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <memory>

class Graph_interface
{
public:
    virtual ~Graph_interface() = default;

    virtual bool is_valid() const = 0;
    virtual int node_count() const = 0;
    virtual bool has_supported_types(
        std::set<hipdnn_sdk::data_objects::NodeAttributes> supported_attributes) const
        = 0;
};

class Graph_wrapper : public Graph_interface
{
public:
    explicit Graph_wrapper(const uint8_t* buffer, size_t size)
    {
        if(buffer)
        {
            flatbuffers::Verifier verifier(buffer, size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::Graph>())
            {
                _graph = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::Graph& get_graph() const
    {
        return *_graph;
    }

    bool is_valid() const override
    {
        return _graph != nullptr;
    }

    int node_count() const override
    {
        return _graph ? _graph->nodes()->size() : 0;
    }

    bool has_supported_types(
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

private:
    const hipdnn_sdk::data_objects::Graph* _graph = nullptr;
};