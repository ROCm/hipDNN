// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ostream>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/logging/logger.hpp>

namespace hipdnn_sdk::data_objects
{
inline const char* to_string(hipdnn_sdk::data_objects::NodeAttributes attributes)
{
    return hipdnn_sdk::data_objects::EnumNameNodeAttributes(attributes);
}

inline const char* to_string(hipdnn_sdk::data_objects::DataType data_type)
{
    return hipdnn_sdk::data_objects::EnumNameDataType(data_type);
}
}

inline std::ostream& operator<<(std::ostream& os,
                                hipdnn_sdk::data_objects::NodeAttributes attributes)
{
    return os << hipdnn_sdk::data_objects::EnumNameNodeAttributes(attributes);
}

inline std::ostream& operator<<(std::ostream& os, hipdnn_sdk::data_objects::DataType data_type)
{
    return os << hipdnn_sdk::data_objects::EnumNameDataType(data_type);
}

template <>
struct fmt::formatter<hipdnn_sdk::data_objects::NodeAttributes> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnn_sdk::data_objects::NodeAttributes attributes, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(
            hipdnn_sdk::data_objects::EnumNameNodeAttributes(attributes), ctx);
    }
};

template <>
struct fmt::formatter<hipdnn_sdk::data_objects::DataType> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnn_sdk::data_objects::DataType data_type, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(
            hipdnn_sdk::data_objects::EnumNameDataType(data_type), ctx);
    }
};
