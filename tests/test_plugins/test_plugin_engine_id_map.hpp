#pragma once

#include <cstdint>

namespace hipdnn_tests::plugin_constants
{
template <class T>
constexpr int64_t engine_id() = delete;
} // namespace hipdnn_tests::plugin_constants

#define HIPDNN_MAP_TO_ID(Class_name, id)      \
    class Class_name;                         \
    namespace hipdnn_tests::plugin_constants  \
    {                                         \
    template <>                               \
    constexpr int64_t engine_id<Class_name>() \
    {                                         \
        return id;                            \
    };                                        \
    }

HIPDNN_MAP_TO_ID(Good_plugin, -2);
HIPDNN_MAP_TO_ID(Good_default_plugin, -3);
HIPDNN_MAP_TO_ID(No_applicable_engines_plugin, -4);
HIPDNN_MAP_TO_ID(Execute_fails_plugin, -5);
HIPDNN_MAP_TO_ID(Duplicate_id_a_plugin, -6);
HIPDNN_MAP_TO_ID(Duplicate_id_b_plugin, -6);
