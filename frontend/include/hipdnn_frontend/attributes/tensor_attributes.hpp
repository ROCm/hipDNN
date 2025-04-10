// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "../types.hpp"
#include <string>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
class Tensor_attributes
{
public:
    int64_t get_uid() const
    {
        return _uid;
    }

    const std::string& get_name() const
    {
        return _name;
    }

    DataType_t get_data_type() const
    {
        return _data_type;
    }

    const std::vector<int64_t>& get_stride() const
    {
        return _stride;
    }

    const std::vector<int64_t>& get_dim() const
    {
        return _dim;
    }

    int64_t get_volume() const
    {
        int64_t volume = 1;
        for(const auto& d : _dim)
        {
            volume *= d;
        }
        return volume;
    }

    bool get_is_virtual() const
    {
        return _is_virtual;
    }

    bool has_uid() const
    {
        return _uid_set;
    }

    Tensor_attributes& set_uid(int64_t uid)
    {
        _uid     = uid;
        _uid_set = true;
        return *this;
    }

    Tensor_attributes& set_name(const std::string& name)
    {
        _name = name;
        return *this;
    }

    Tensor_attributes& set_data_type(DataType_t data_type)
    {
        _data_type = data_type;
        return *this;
    }

    Tensor_attributes& set_stride(const std::vector<int64_t>& stride)
    {
        _stride = stride;
        return *this;
    }

    Tensor_attributes& set_dim(const std::vector<int64_t>& dim)
    {
        _dim = dim;
        return *this;
    }

    Tensor_attributes& set_is_virtual(bool is_virtual)
    {
        _is_virtual = is_virtual;
        return *this;
    }

    Tensor_attributes& set_ouput(bool output)
    {
        return set_is_virtual(!output);
    }

    Tensor_attributes& clear_uid()
    {
        _uid     = 0;
        _uid_set = false;
        return *this;
    }

private:
    int64_t              _uid     = 0;
    bool                 _uid_set = false;
    std::string          _name;
    DataType_t           _data_type = DataType_t::NOT_SET;
    std::vector<int64_t> _stride;
    std::vector<int64_t> _dim;
    bool                 _is_virtual = false;
};

}
}