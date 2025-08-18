// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../test_plugins/test_plugin_constants.hpp"
#include "hipdnn_backend.h"
#include "test_util.hpp"
#include <array>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/platform_utils.hpp>
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
    hipdnnBackendDescriptor_t execution_plan = nullptr;
    hipdnnBackendDescriptor_t variant_pack = nullptr;

    hipdnnStatus_t status = hipdnnBackendExecute(handle, execution_plan, variant_pack);

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
    hipdnnBackendAttributeName_t attribute_name = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t attribute_type = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t requested_element_count = 0;
    int64_t element_count = 0;
    void* array_of_elements = nullptr;

    hipdnnStatus_t status = hipdnnBackendGetAttribute(descriptor,
                                                      attribute_name,
                                                      attribute_type,
                                                      requested_element_count,
                                                      &element_count,
                                                      array_of_elements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, SetAttribute)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    hipdnnBackendAttributeName_t attribute_name = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t attribute_type = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t element_count = 0;
    void* array_of_elements = nullptr;

    hipdnnStatus_t status = hipdnnBackendSetAttribute(
        descriptor, attribute_name, attribute_type, element_count, array_of_elements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, WillSetBackendGraphCorrectly)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
        tensor_attributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto graph
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "Test GRAPH!",
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      &tensor_attributes,
                                                      &nodes);
    builder.Finish(graph);
    flatbuffers::DetachedBuffer serialized_graph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &descriptor, serialized_graph.data(), serialized_graph.size());

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

TEST(HipDNNBackendTest, SetPluginPathsExt_Success)
{
    using namespace hipdnn_tests::plugin_constants;
    std::string plugin_dir_str = PLUGIN_DIR.string();
    std::array<const char*, 3> paths = {plugin_dir_str.c_str(), "./", "../directory/"};

    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, SetPluginPathsExt_InvalidAndValidNullPointerCorrectness)
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

    auto loaded_plugins = test_util::get_loaded_plugins(handle);
    EXPECT_EQ(loaded_plugins.size(), 0);

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, SetPluginPathsExt_FailsOnNullStringInList)
{
    std::array<const char*, 2> paths = {"./valid/path.so", nullptr};

    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, PluginPathsExt_FailsWithIneligibleHandle)
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

    size_t num_plugins = 0;
    size_t max_path_length = 0;
    status = hipdnnGetLoadedEnginePluginPaths_ext(nullptr, &num_plugins, nullptr, &max_path_length);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(HipDNNBackendTest, GetLoadedPluginPaths_LoadsDefault)
{
    hipdnnStatus_t status
        = hipdnnSetEnginePluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loaded_plugins = test_util::get_loaded_plugins(handle);

    fs::path expected_plugin_path = fs::path("../../backend/src/hipdnn_plugins/engines")
                                    / get_library_name("test_good_default_plugin");

    EXPECT_TRUE(test_util::is_plugin_loaded(loaded_plugins, expected_plugin_path.string()));
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, GetLoadedPluginPaths_AdditiveLoadsBothDefaultAndCustom)
{
    std::string plugin_dir_str = PLUGIN_DIR.string();
    const std::array<const char*, 1> paths = {plugin_dir_str.c_str()};
    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loaded_plugins = test_util::get_loaded_plugins(handle);
    EXPECT_GE(loaded_plugins.size(), 2);

    auto default_plugin_path = fs::path("../../backend/src/hipdnn_plugins/engines")
                               / get_library_name("test_good_default_plugin");
    auto test_plugin_path = PLUGIN_DIR / get_library_name(test_good_plugin_name);

    EXPECT_TRUE(test_util::is_plugin_loaded(loaded_plugins, default_plugin_path.string()));
    EXPECT_TRUE(test_util::is_plugin_loaded(loaded_plugins, test_plugin_path.string()));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, GetLoadedPluginPaths_AbsoluteLoadsOnlyCustom)
{
    auto& plugin_file_path = test_good_plugin_path();
    const std::array<const char*, 1> paths = {plugin_file_path.c_str()};
    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loaded_plugins = test_util::get_loaded_plugins(handle);
    EXPECT_EQ(loaded_plugins.size(), 1);

    auto default_plugin_path = fs::path("backend/src/hipdnn_plugins/engines")
                               / get_library_name("test_good_default_plugin");

    EXPECT_FALSE(test_util::is_plugin_loaded(loaded_plugins, default_plugin_path.string()));
    EXPECT_TRUE(test_util::is_plugin_loaded(loaded_plugins, plugin_file_path));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
