// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "graph_attributes.hpp"
#include <hipdnn_frontend/types.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <type_traits>
#include <flatbuffers/flatbuffers.h>

namespace hipdnn_frontend
{
namespace graph
{
class Tensor_attributes
{
public:
    using ValueVariant = std::variant<std::monostate, float, uint16_t, uint8_t, int32_t>;

    bool has_value() const { return !std::holds_alternative<std::monostate>(_value); }

    template<typename T>
    std::optional<T> get_value() const {
        if (auto p = std::get_if<T>(&_value)){
            return *p;
        }
        return std::nullopt;
    }

    template<typename T>
    Tensor_attributes& set_value(T v)
    {
        static_assert(
            std::disjunction_v<
                std::is_same<T,float>,
                std::is_same<T,uint16_t>,
                std::is_same<T,uint8_t>,
                std::is_same<T,int32_t>>,
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
        // using FB = hipdnn_sdk::data_objects;

        auto [value_type, value_offset] = [&]() {
            return std::visit([&](auto &&arg) -> std::pair<hipdnn_sdk::data_objects::Tensor_Value, flatbuffers::Offset<void>> {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, float>){
                    return { hipdnn_sdk::data_objects::Tensor_Value_FValue,
                            hipdnn_sdk::data_objects::CreateFValue(builder, arg).Union() };
                }
                else if constexpr (std::is_same_v<T, uint16_t>){
                    return { hipdnn_sdk::data_objects::Tensor_Value_HValue,
                            hipdnn_sdk::data_objects::CreateHValue(builder, arg).Union() };
                }
                else if constexpr (std::is_same_v<T, uint8_t>)
                {    return { hipdnn_sdk::data_objects::Tensor_Value_UValue,
                            hipdnn_sdk::data_objects::CreateUValue(builder, arg).Union() };
                }
                else if constexpr (std::is_same_v<T, int32_t>)
                {    return { hipdnn_sdk::data_objects::Tensor_Value_IValue,
                            hipdnn_sdk::data_objects::CreateIValue(builder, arg).Union() };
                }
                else
                {
                    return { hipdnn_sdk::data_objects::Tensor_Value_NONE, 0 };
                }
            }, _value);
        }();

        return CreateTensorAttributesDirect(
            builder, _uid, _name.c_str(), to_sdk_type(_data_type), &_stride, &_dim, _is_virtual,
            value_type,
            value_offset);
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