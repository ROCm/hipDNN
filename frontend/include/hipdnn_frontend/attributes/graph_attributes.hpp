// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "../types.hpp"

namespace hipdnn_frontend
{
namespace graph
{
class Graph_attributes
{
public:
    const std::string& get_name() const
    {
        return _name;
    }
    DataType_t get_compute_data_type() const
    {
        return _compute_type;
    }
    DataType_t get_intermediate_data_type() const
    {
        return _intermediate_type;
    }
    DataType_t get_io_data_type() const
    {
        return _io_type;
    }

    Graph_attributes& set_compute_data_type(DataType_t compute_type)
    {
        _compute_type = compute_type;
        return *this;
    }
    Graph_attributes& set_intermediate_data_type(DataType_t intermediate_type)
    {
        _intermediate_type = intermediate_type;
        return *this;
    }
    Graph_attributes& set_io_data_type(DataType_t io_type)
    {
        _io_type = io_type;
        return *this;
    }
    Graph_attributes& set_name(const std::string& name)
    {
        _name = name;
        return *this;
    }

private:
    std::string _name;
    DataType_t  _compute_type      = DataType_t::NOT_SET;
    DataType_t  _intermediate_type = DataType_t::NOT_SET;
    DataType_t  _io_type           = DataType_t::NOT_SET;
};
}
}