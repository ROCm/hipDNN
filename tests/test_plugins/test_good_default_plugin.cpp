// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_plugin_common.hpp"
#include "test_plugin_engine_id_map.hpp"
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char
    hipdnn_plugin::Plugin_last_error_manager::last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class Good_default_plugin : public Test_plugin_base
{
public:
    const char* get_plugin_name() const override
    {
        return "test_good_default_plugin";
    }
    const char* get_plugin_version() const override
    {
        return "1.0.0";
    }
    int64_t get_engine_id() const override
    {
        return hipdnn_tests::plugin_constants::engine_id<Good_default_plugin>();
    }
    uint32_t get_num_engines() const override
    {
        return 1;
    }
    uint32_t get_num_applicable_engines() const override
    {
        return 1;
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initialize_plugin()
{
    Test_plugin_base::set_instance(std::make_unique<Good_default_plugin>());
}

// Register all API functions
REGISTER_TEST_PLUGIN_API()
