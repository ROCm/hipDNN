// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "test_util.hpp"
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <vector>

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
    std::array<const char*, 3> paths = {"../test_plugins/test_good_plugin", "./", "../directory/"};

    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, SetPluginPathsExt_FailsOnNullPointer)
{
    hipdnnStatus_t status
        = hipdnnSetEnginePluginPaths_ext(1, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    status = hipdnnSetEnginePluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, SetPluginPathsExt_FailsOnNullStringInList)
{
    std::array<const char*, 2> paths = {"./valid/path.so", nullptr};

    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

class Hipdnn_backend_plugin_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const std::array<const char*, 0> paths = {};
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE);
        const std::filesystem::path dest_dir = "../../backend/src/hipdnn_plugins/engines/";
        const std::filesystem::path source_file = "../test_plugins/libtest_good_plugin.so";
        const std::filesystem::path dest_file_with_new_name
            = dest_dir / "libtest_good_default_plugin.so";

        std::filesystem::create_directories(dest_dir);
        std::filesystem::copy(source_file,
                              dest_file_with_new_name,
                              std::filesystem::copy_options::overwrite_existing);
    }
    void TearDown() override
    {
        hipdnnSetEnginePluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
        std::filesystem::remove_all("hipdnn_plugins");
    }
};

TEST_F(Hipdnn_backend_plugin_test, GetLoadedPluginPaths_LoadsDefault)
{
    hipdnnHandle_t handle = nullptr;
    auto status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    const auto loaded_plugins = test_util::get_loaded_plugins();
    EXPECT_TRUE(test_util::is_plugin_loaded(loaded_plugins, "libtest_good_default_plugin.so"))
        << "The default plugin was not loaded.";

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST_F(Hipdnn_backend_plugin_test, GetLoadedPluginPaths_AdditiveLoadsBothDefaultAndCustom)
{
    const std::array<const char*, 1> paths = {"../../tests/test_plugins/"};
    auto status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    const auto loaded_plugins = test_util::get_loaded_plugins();
    EXPECT_GE(loaded_plugins.size(), 2);
    EXPECT_TRUE(test_util::is_plugin_loaded(loaded_plugins, "libtest_good_default_plugin.so"));
    EXPECT_TRUE(test_util::is_plugin_loaded(loaded_plugins, "libtest_good_plugin.so"));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST_F(Hipdnn_backend_plugin_test, GetLoadedPluginPaths_AbsoluteLoadsOnlyCustom)
{
    const std::array<const char*, 1> paths = {"../../tests/test_plugins/"};
    auto status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    const auto loaded_plugins = test_util::get_loaded_plugins();
    EXPECT_EQ(loaded_plugins.size(), 1);
    EXPECT_FALSE(test_util::is_plugin_loaded(loaded_plugins, "libtest_good_default_plugin.so"));
    EXPECT_TRUE(test_util::is_plugin_loaded(loaded_plugins, "libtest_good_plugin.so"));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
