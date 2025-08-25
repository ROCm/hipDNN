// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "descriptors/backend_descriptor.hpp"
#include "hipdnn_backend.h"
#include "hipdnn_sdk/plugin/engine_plugin_api.h"
#include "hipdnn_sdk/plugin/plugin_api.h"
#include "hipdnn_sdk/utilities/platform_utils.hpp"
#include "test_util.hpp"
#include <hipdnn_backend_attribute_name.h>
#include <hipdnn_backend_attribute_type.h>
#include <hipdnn_backend_heuristic_type.h>
#include <hipdnn_sdk/test_utilities/temp_directory.hpp>
#include <test_plugins/test_plugin_constants.hpp>

#include <gtest/gtest.h>

class Plugin_loading_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engine_config = nullptr;
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;
    hipdnnBackendDescriptor_t _heuristic_descriptor = nullptr;
    hipdnnHandle_t _handle = nullptr;

    void SetUp() override {}

    void TearDown() override
    {
        if(_engine_config != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine_config), HIPDNN_STATUS_SUCCESS);
            _engine_config = nullptr;
        }
        if(_engine != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine), HIPDNN_STATUS_SUCCESS);
            _engine = nullptr;
        }
        if(_graph != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_graph), HIPDNN_STATUS_SUCCESS);
            _graph = nullptr;
        }
        if(_heuristic_descriptor != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_heuristic_descriptor), HIPDNN_STATUS_SUCCESS);
            _heuristic_descriptor = nullptr;
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }
};

void create_heuristic_descriptor(hipdnnBackendDescriptor_t* heuristic_descriptor,
                                 hipdnnBackendDescriptor_t* graph,
                                 bool finalize = false)
{
    EXPECT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, heuristic_descriptor),
        HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendSetAttribute(*heuristic_descriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        graph),
              HIPDNN_STATUS_SUCCESS);

    hipdnnBackendHeurMode_t backend_modes = HIPDNN_HEUR_MODE_FALLBACK;

    EXPECT_EQ(hipdnnBackendSetAttribute(*heuristic_descriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &backend_modes),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        EXPECT_EQ(hipdnnBackendFinalize(*heuristic_descriptor), HIPDNN_STATUS_SUCCESS);
    }
}

TEST_F(Plugin_loading_tests, EmptyPluginPath)
{
    Temp_directory plugin_dir("empty_plugins");
    auto plugin_path = plugin_dir.path().string();
    const std::array<const char*, 1> paths = {plugin_path.c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engine_config),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engine_config, nullptr);

    test_util::create_test_graph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    create_heuristic_descriptor(&_heuristic_descriptor, &_graph, true);

    int64_t available_engine_count = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristic_descriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &available_engine_count,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(available_engine_count, 0);
}

TEST_F(Plugin_loading_tests, NoPluginsSupportGraph)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::test_no_applicable_engines_plugin_path().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engine_config),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engine_config, nullptr);

    test_util::create_test_graph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    create_heuristic_descriptor(&_heuristic_descriptor, &_graph, true);

    int64_t available_engine_count = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristic_descriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &available_engine_count,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(available_engine_count, 0);
}

TEST_F(Plugin_loading_tests, IncorrectEngineID)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::test_no_applicable_engines_plugin_path().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engine_config),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engine_config, nullptr);

    test_util::create_test_graph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    test_util::create_test_engine(&_engine, &_graph, _handle, -193489);

    ASSERT_EQ(hipdnnBackendFinalize(_engine), HIPDNN_STATUS_BAD_PARAM);

    constexpr size_t buffer_size = 512;
    std::array<char, buffer_size> buffer;
    hipdnnGetLastErrorString(buffer.data(), buffer_size);

    ASSERT_EQ(
        std::string{buffer.data()},
        "Engine_descriptor::finalize() failed: Engine id is not in a valid range of engine IDs");
}

TEST_F(Plugin_loading_tests, DuplicateEngineIds)
{
    const std::array<const char*, 2> paths
        = {hipdnn_tests::plugin_constants::test_duplicate_id_a_plugin_path().c_str(),
           hipdnn_tests::plugin_constants::test_duplicate_id_b_plugin_path().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    // TODO: Warning is logged, but we don't have means of querying the last warning

    EXPECT_EQ(test_util::get_loaded_plugins(_handle).size(), 1);
}

TEST_F(Plugin_loading_tests, IncompleteAPI)
{
    using namespace hipdnn_sdk::utilities;
    using namespace hipdnn_tests::plugin_constants;

    const std::array<const char*, 1> paths = {test_incomplete_api_plugin_path().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    // TODO: Warning is logged, but we don't have means of querying the last warning

    EXPECT_EQ(test_util::get_loaded_plugins(_handle).size(), 0);
}

TEST_F(Plugin_loading_tests, MultiplePluginsOneApplicableEngine)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::test_no_applicable_engines_plugin_path().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engine_config),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engine_config, nullptr);

    test_util::create_test_graph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    create_heuristic_descriptor(&_heuristic_descriptor, &_graph, true);

    int64_t available_engine_count = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristic_descriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &available_engine_count,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(available_engine_count, 1);
}

TEST_F(Plugin_loading_tests, MultiplePluginsMultipleApplicableEngines)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::test_good_plugin_path().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engine_config),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engine_config, nullptr);

    test_util::create_test_graph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    create_heuristic_descriptor(&_heuristic_descriptor, &_graph, true);

    int64_t available_engine_count = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristic_descriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &available_engine_count,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(available_engine_count, 2);
}
