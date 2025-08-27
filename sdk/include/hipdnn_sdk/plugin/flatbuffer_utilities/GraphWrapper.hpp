// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>

namespace hipdnn_plugin
{

class IGraph
{
public:
    virtual ~IGraph() = default;

    virtual const hipdnn_sdk::data_objects::Graph& getGraph() const = 0;
    virtual bool isValid() const = 0;
    virtual uint32_t nodeCount() const = 0;
    virtual bool hasOnlySupportedAttributes(
        std::set<hipdnn_sdk::data_objects::NodeAttributes> supportedAttributes) const
        = 0;
    virtual const hipdnn_sdk::data_objects::Node& getNode(uint32_t index) const = 0;
    virtual const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        getTensorMap() const
        = 0;
};

class GraphWrapper : public IGraph
{
public:
    explicit GraphWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::Graph>())
            {
                _shallowGraph = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::Graph& getGraph() const override
    {
        throwIfNotValid();
        return *_shallowGraph;
    }

    bool isValid() const override
    {
        return _shallowGraph != nullptr;
    }

    uint32_t nodeCount() const override
    {
        throwIfNotValid();
        return _shallowGraph->nodes()->size();
    }

    bool hasOnlySupportedAttributes(
        std::set<hipdnn_sdk::data_objects::NodeAttributes> supportedAttributes) const override
    {
        throwIfNotValid();

        // NOLINTNEXTLINE(readability-use-anyofallof)
        for(const auto node : *_shallowGraph->nodes())
        {
            if(!supportedAttributes.contains(node->attributes_type()))
            {
                return false;
            }
        }
        return true;
    }

    const hipdnn_sdk::data_objects::Node& getNode(uint32_t index) const override
    {
        throwIfNotValid();

        if(index >= _shallowGraph->nodes()->size())
        {
            throw std::out_of_range("Index out of range for graph nodes");
        }

        return *_shallowGraph->nodes()->Get(index);
    }

    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        getTensorMap() const override
    {
        throwIfNotValid();

        if(!_tensorMap.empty())
        {
            return _tensorMap;
        }

        for(const auto tensor : *_shallowGraph->tensors())
        {
            _tensorMap[tensor->uid()] = tensor;
        }

        return _tensorMap;
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Graph is not valid");
        }
    }

    // Pointer to the flatbuffer representation of the graph. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_sdk::data_objects::Graph* _shallowGraph = nullptr;

    //lazy init state;
    mutable std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>
        _tensorMap;
};

}
