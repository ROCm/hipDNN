// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../test_plugins/TestPluginConstants.hpp"
#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <array>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <vector>

using namespace hipdnn_tests::plugin_constants;
using namespace hipdnn_sdk::utilities;
namespace fs = std::filesystem;

TEST(HipDNNBackendTest, WillCreateDestroyGraphDescriptorSuccessfully)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, WillNotCreateDescriptorIfPassedNullptr)
{
    hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, nullptr);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, WillNotCreateDescriptorIfTypeNotSupported)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status = hipdnnBackendCreateDescriptor(HIPDNN_INVALID_TYPE, &descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST(HipDNNBackendTest, WontDestroyDescriptorIfNull)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status = hipdnnBackendDestroyDescriptor(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, CreateHandleFailsIfHandlePtrIsNull)
{
    hipdnnStatus_t status = hipdnnCreate(nullptr);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, Execute)
{
    hipdnnHandle_t handle = nullptr;
    hipdnnBackendDescriptor_t executionPlan = nullptr;
    hipdnnBackendDescriptor_t variantPack = nullptr;

    hipdnnStatus_t status = hipdnnBackendExecute(handle, executionPlan, variantPack);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, Finalize)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status = hipdnnBackendFinalize(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, GetAttribute)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    hipdnnBackendAttributeName_t attributeName = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t attributeType = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t requestedElementCount = 0;
    int64_t elementCount = 0;
    void* arrayOfElements = nullptr;

    hipdnnStatus_t status = hipdnnBackendGetAttribute(descriptor,
                                                      attributeName,
                                                      attributeType,
                                                      requestedElementCount,
                                                      &elementCount,
                                                      arrayOfElements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, SetAttribute)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    hipdnnBackendAttributeName_t attributeName = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t attributeType = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t elementCount = 0;
    void* arrayOfElements = nullptr;

    hipdnnStatus_t status = hipdnnBackendSetAttribute(
        descriptor, attributeName, attributeType, elementCount, arrayOfElements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, WillSetBackendGraphCorrectly)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto graph
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "Test GRAPH!",
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graph);
    flatbuffers::DetachedBuffer serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &descriptor, serializedGraph.data(), serializedGraph.size());

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendSetAttribute(
        descriptor, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendFinalize(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, WillFailToFinalizeInvalidGraph)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendFinalize(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    status = hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, WillFailToCreateGraphIfGraphIsNull)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(&descriptor, nullptr, 0);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(descriptor, nullptr);
}

TEST(HipDNNBackendTest, SetPluginPathsExtSuccess)
{
    using namespace hipdnn_tests::plugin_constants;
    std::string pluginDirStr = PLUGIN_DIR.string();
    std::array<const char*, 3> paths = {pluginDirStr.c_str(), "./", "../directory/"};

    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, SetPluginPathsExtInvalidAndValidNullPointerCorrectness)
{
    hipdnnStatus_t status
        = hipdnnSetEnginePluginPaths_ext(1, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    status = hipdnnSetEnginePluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loadedPlugins = test_util::getLoadedPlugins(handle);
    EXPECT_EQ(loadedPlugins.size(), 0);

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, SetPluginPathsExtFailsOnNullStringInList)
{
    std::array<const char*, 2> paths = {"./valid/path.so", nullptr};

    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, PluginPathsExtFailsWithIneligibleHandle)
{
    hipdnnHandle_t handle = nullptr;
    hipdnnStatus_t status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    const std::array<const char*, 1> paths = {"./fake/path"};
    status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    size_t numPlugins = 0;
    size_t maxPathLength = 0;
    status = hipdnnGetLoadedEnginePluginPaths_ext(nullptr, &numPlugins, nullptr, &maxPathLength);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, GetLoadedPluginPathsLoadsDefault)
{
    hipdnnStatus_t status
        = hipdnnSetEnginePluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loadedPlugins = test_util::getLoadedPlugins(handle);

    fs::path expectedPluginPath = fs::path("../../backend/src/hipdnn_plugins/engines")
                                  / getLibraryName("test_good_default_plugin");

    EXPECT_TRUE(test_util::isPluginLoaded(loadedPlugins, expectedPluginPath.string()));
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, GetLoadedPluginPathsAdditiveLoadsBothDefaultAndCustom)
{
    std::string pluginDirStr = PLUGIN_DIR.string();
    const std::array<const char*, 1> paths = {pluginDirStr.c_str()};
    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loadedPlugins = test_util::getLoadedPlugins(handle);
    EXPECT_GE(loadedPlugins.size(), 2);

    auto defaultPluginPath = fs::path("../../backend/src/hipdnn_plugins/engines")
                             / getLibraryName("test_good_default_plugin");
    auto testPluginPath = PLUGIN_DIR / getLibraryName(TEST_GOOD_PLUGIN_NAME);

    EXPECT_TRUE(test_util::isPluginLoaded(loadedPlugins, defaultPluginPath.string()));
    EXPECT_TRUE(test_util::isPluginLoaded(loadedPlugins, testPluginPath.string()));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, GetLoadedPluginPathsAbsoluteLoadsOnlyCustom)
{
    auto& pluginFilePath = testGoodPluginPath();
    const std::array<const char*, 1> paths = {pluginFilePath.c_str()};
    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loadedPlugins = test_util::getLoadedPlugins(handle);
    EXPECT_EQ(loadedPlugins.size(), 1);

    auto defaultPluginPath = fs::path("backend/src/hipdnn_plugins/engines")
                             / getLibraryName("test_good_default_plugin");

    EXPECT_FALSE(test_util::isPluginLoaded(loadedPlugins, defaultPluginPath.string()));
    EXPECT_TRUE(test_util::isPluginLoaded(loadedPlugins, pluginFilePath));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
