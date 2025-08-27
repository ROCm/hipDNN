// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestUtil.hpp"
#include "descriptors/BackendDescriptor.hpp"
#include "hipdnn_backend.h"
#include "hipdnn_sdk/plugin/EnginePluginApi.h"
#include "hipdnn_sdk/plugin/PluginApi.h"
#include "hipdnn_sdk/utilities/PlatformUtils.hpp"
#include <HipdnnBackendAttributeName.h>
#include <HipdnnBackendAttributeType.h>
#include <HipdnnBackendHeuristicType.h>
#include <hipdnn_sdk/test_utilities/TempDirectory.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <gtest/gtest.h>

class PluginLoadingTests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engineConfig = nullptr;
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;
    hipdnnBackendDescriptor_t _heuristicDescriptor = nullptr;
    hipdnnHandle_t _handle = nullptr;

    void SetUp() override {}

    void TearDown() override
    {
        if(_engineConfig != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engineConfig), HIPDNN_STATUS_SUCCESS);
            _engineConfig = nullptr;
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
        if(_heuristicDescriptor != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_heuristicDescriptor), HIPDNN_STATUS_SUCCESS);
            _heuristicDescriptor = nullptr;
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }
};

void createHeuristicDescriptor(hipdnnBackendDescriptor_t* heuristicDescriptor,
                               hipdnnBackendDescriptor_t* graph,
                               bool finalize = false)
{
    EXPECT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, heuristicDescriptor),
        HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendSetAttribute(*heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        graph),
              HIPDNN_STATUS_SUCCESS);

    auto backendModes = HIPDNN_HEUR_MODE_FALLBACK;

    EXPECT_EQ(hipdnnBackendSetAttribute(*heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &backendModes),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        EXPECT_EQ(hipdnnBackendFinalize(*heuristicDescriptor), HIPDNN_STATUS_SUCCESS);
    }
}

TEST_F(PluginLoadingTests, EmptyPluginPath)
{
    TempDirectory pluginDir("empty_plugins");
    auto pluginPath = pluginDir.path().string();
    const std::array<const char*, 1> paths = {pluginPath.c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    createHeuristicDescriptor(&_heuristicDescriptor, &_graph, true);

    auto availableEngineCount = int64_t{-1};
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &availableEngineCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(availableEngineCount, 0);
}

TEST_F(PluginLoadingTests, NoPluginsSupportGraph)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testNoApplicableEnginesPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    createHeuristicDescriptor(&_heuristicDescriptor, &_graph, true);

    auto availableEngineCount = int64_t{-1};
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &availableEngineCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(availableEngineCount, 0);
}

TEST_F(PluginLoadingTests, IncorrectEngineID)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testNoApplicableEnginesPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    test_util::createTestEngine(&_engine, &_graph, _handle, -193489);

    ASSERT_EQ(hipdnnBackendFinalize(_engine), HIPDNN_STATUS_BAD_PARAM);

    constexpr size_t BUFFER_SIZE = 512;
    std::array<char, BUFFER_SIZE> buffer;
    hipdnnGetLastErrorString(buffer.data(), BUFFER_SIZE);

    ASSERT_EQ(
        std::string{buffer.data()},
        "EngineDescriptor::finalize() failed: Engine id is not in a valid range of engine IDs");
}

TEST_F(PluginLoadingTests, DuplicateEngineIds)
{
    const std::array<const char*, 2> paths
        = {hipdnn_tests::plugin_constants::testDuplicateIdAPluginPath().c_str(),
           hipdnn_tests::plugin_constants::testDuplicateIdBPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    // TODO: Warning is logged, but we don't have means of querying the last warning

    EXPECT_EQ(test_util::getLoadedPlugins(_handle).size(), 1);
}

TEST_F(PluginLoadingTests, IncompleteAPI)
{
    using namespace hipdnn_sdk::utilities;
    using namespace hipdnn_tests::plugin_constants;

    const std::array<const char*, 1> paths = {testIncompleteApiPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    // TODO: Warning is logged, but we don't have means of querying the last warning

    EXPECT_EQ(test_util::getLoadedPlugins(_handle).size(), 0);
}

TEST_F(PluginLoadingTests, MultiplePluginsOneApplicableEngine)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testNoApplicableEnginesPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    createHeuristicDescriptor(&_heuristicDescriptor, &_graph, true);

    auto availableEngineCount = int64_t{-1};
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &availableEngineCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(availableEngineCount, 1);
}

TEST_F(PluginLoadingTests, MultiplePluginsMultipleApplicableEngines)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    createHeuristicDescriptor(&_heuristicDescriptor, &_graph, true);

    auto availableEngineCount = int64_t{-1};
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &availableEngineCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(availableEngineCount, 2);
}
