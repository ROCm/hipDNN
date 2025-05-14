// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "graph_attributes.hpp"
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/types.hpp>
#include <string>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
// Any class extending AttributesCRTP must have an inputs & outputs map.
// The map needs to have TensorAttributes as the value.
// AttributesCRTP uses these maps to set the tensor data types.
template <typename DerivedT>
class AttributesCRTP // NOLINT
{
private:
    DerivedT& self()
    {
        return static_cast<DerivedT&>(*this);
    }
    const DerivedT& self() const
    {
        return static_cast<const DerivedT&>(*this);
    }

public:
    std::string name;

    DerivedT& set_name(const std::string& name_)
    {
        name = name_;
        return self();
    }

    const std::string& get_name() const
    {
        return name;
    }

    error_t fill_from_graph_attributes(const Graph_attributes& graph_attributes)
    {
        for(auto& [_, tensor] : self().inputs)
        {
            if(tensor)
            {
                tensor->set_from_graph_attributes(graph_attributes);
            }
        }

        for(auto& [_, tensor] : self().outputs)
        {
            if(tensor)
            {
                tensor->set_from_graph_attributes(graph_attributes);
            }
        }

        return {};
    }
};
}
}