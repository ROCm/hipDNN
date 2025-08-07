// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "graph_attributes.hpp"
#include <flatbuffers/flatbuffers.h>
#include <hipdnn_frontend/types.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
class Tensor_attributes
{
public:
    using ValueVariant = std::variant<std::monostate, double, float, uint16_t, uint8_t, int32_t>;

    bool has_value() const
    {
        return !std::holds_alternative<std::monostate>(_value);
    }

    template <typename T>
    std::optional<T> get_value() const
    {
        if(auto p = std::get_if<T>(&_value))
        {
            return *p;
        }
        return std::nullopt;
    }

    template <typename T>
    Tensor_attributes& set_value(T v)
    {
        static_assert(std::disjunction_v<std::is_same<T, float>,
                                         std::is_same<T, double>,
                                         std::is_same<T, uint16_t>,
                                         std::is_same<T, uint8_t>,
                                         std::is_same<T, int32_t>>,
                      "Unsupported type for Tensor_attributes::set_value");
        _value = v;
        return *this;
    }

    Tensor_attributes& clear_value()
    {
        _value = {};
        return *this;
    }

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
        _uid = uid;
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

    Tensor_attributes& set_output(bool output)
    {
        return set_is_virtual(!output);
    }

    Tensor_attributes& clear_uid()
    {
        _uid = 0;
        _uid_set = false;
        return *this;
    }

    Tensor_attributes& set_from_graph_attributes(const Graph_attributes& graph_attributes)
    {
        if(_data_type == DataType_t::NOT_SET)
        {
            if(_is_virtual)
            {
                _data_type = graph_attributes.get_intermediate_data_type();
            }
            else
            {
                _data_type = graph_attributes.get_io_data_type();
            }
        }

        return *this;
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const
    {
        auto result = std::visit(
            [&](auto&& arg)
                -> std::pair<hipdnn_sdk::data_objects::TensorValue, flatbuffers::Offset<void>> {
                using T = std::decay_t<decltype(arg)>;
                if constexpr(std::is_same_v<T, float>)
                {
                    hipdnn_sdk::data_objects::Float32Value float_val(arg);
                    return {hipdnn_sdk::data_objects::TensorValue_Float32Value,
                            builder.CreateStruct(float_val).Union()};
                }
                else if constexpr(std::is_same_v<T, double>)
                {
                    hipdnn_sdk::data_objects::Float64Value double_val(arg);
                    return {hipdnn_sdk::data_objects::TensorValue_Float64Value,
                            builder.CreateStruct(double_val).Union()};
                }
                else if constexpr(std::is_same_v<T, uint16_t>)
                {
                    hipdnn_sdk::data_objects::Float16Value half_val(arg);
                    return {hipdnn_sdk::data_objects::TensorValue_Float16Value,
                            builder.CreateStruct(half_val).Union()};
                }
                else if constexpr(std::is_same_v<T, uint8_t>)
                {
                    hipdnn_sdk::data_objects::Float8Value uint8_val(arg);
                    return {hipdnn_sdk::data_objects::TensorValue_Float8Value,
                            builder.CreateStruct(uint8_val).Union()};
                }
                else if constexpr(std::is_same_v<T, int32_t>)
                {
                    hipdnn_sdk::data_objects::Int32Value int32_val(arg);
                    return {hipdnn_sdk::data_objects::TensorValue_Int32Value,
                            builder.CreateStruct(int32_val).Union()};
                }
                else
                {
                    // For std::monostate case
                    return {hipdnn_sdk::data_objects::TensorValue_NONE, 0};
                }
            },
            _value);

        return hipdnn_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                      _uid,
                                                                      _name.c_str(),
                                                                      to_sdk_type(_data_type),
                                                                      &_stride,
                                                                      &_dim,
                                                                      _is_virtual,
                                                                      result.first,
                                                                      result.second);
    }

private:
    int64_t _uid = 0;
    bool _uid_set = false;
    std::string _name;
    DataType_t _data_type = DataType_t::NOT_SET;
    std::vector<int64_t> _stride;
    std::vector<int64_t> _dim;
    bool _is_virtual = false;
    ValueVariant _value;
};

}
}
