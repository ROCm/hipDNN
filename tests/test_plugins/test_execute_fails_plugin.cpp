// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_plugin_common.hpp"

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char
    hipdnn_plugin::Plugin_last_error_manager::last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class Execute_fails_plugin : public Test_plugin_base
{
public:
    const char* get_plugin_name() const override
    {
        return "test_execute_fails_plugin";
    }
    const char* get_plugin_version() const override
    {
        return "1.0.0";
    }
    int64_t get_engine_id() const override
    {
        return -2;
    }
    uint32_t get_num_engines() const override
    {
        return 1;
    }
    uint32_t get_num_applicable_engines() const override
    {
        return 1;
    }

    // Override execute_graph to simulate execution failure
    void execute_graph() const override
    {
        throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                     "Simulated execution failure for testing");
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initialize_plugin()
{
    Test_plugin_base::set_instance(std::make_unique<Execute_fails_plugin>());
}

// Register all API functions
REGISTER_TEST_PLUGIN_API()
