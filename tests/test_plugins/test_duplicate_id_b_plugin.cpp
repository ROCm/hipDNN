// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_plugin_common.hpp"
#include "test_plugin_engine_id_map.hpp"
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char
    hipdnn_plugin::PluginLastErrorManager::_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class DuplicateIdBPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_DuplicateIdBPlugin";
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }
    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<DuplicateIdBPlugin>();
    }
    uint32_t getNumEngines() const override
    {
        return 1;
    }
    uint32_t getNumApplicableEngines() const override
    {
        return 1;
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<DuplicateIdBPlugin>());
}

// Register all API functions
REGISTER_TEST_PLUGIN_API()
