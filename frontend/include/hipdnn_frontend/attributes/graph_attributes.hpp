// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/types.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
{
class GraphAttributes
{
public:
    const std::string& get_name() const
    {
        return _name;
    }
    DataType_t get_compute_data_type() const
    {
        return _computeType;
    }
    DataType_t get_intermediate_data_type() const
    {
        return _intermediateType;
    }
    DataType_t get_io_data_type() const
    {
        return _ioType;
    }

    GraphAttributes& set_compute_data_type(DataType_t computeType)
    {
        _computeType = computeType;
        return *this;
    }
    GraphAttributes& set_intermediate_data_type(DataType_t intermediateType)
    {
        _intermediateType = intermediateType;
        return *this;
    }
    GraphAttributes& set_io_data_type(DataType_t ioType)
    {
        _ioType = ioType;
        return *this;
    }
    GraphAttributes& set_name(const std::string& name)
    {
        _name = name;
        return *this;
    }

private:
    std::string _name;
    DataType_t _computeType = DataType_t::NOT_SET;
    DataType_t _intermediateType = DataType_t::NOT_SET;
    DataType_t _ioType = DataType_t::NOT_SET;
};
typedef GraphAttributes Graph_attributes;
}
}
