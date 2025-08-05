// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "hipdnn_status.h"
#include "test_macros.hpp"
#include "test_util.hpp"

#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

#include <gtest/gtest.h>

class Execution_backend_end_api_tests : public ::testing::Test
{
protected:
    static constexpr int64_t GIDX = -1; // TODO hardcode for now
    hipdnnBackendDescriptor_t _plan = nullptr;
    hipdnnHandle_t _handle = nullptr;
    hipdnnBackendDescriptor_t _engine_config = nullptr;
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _graph_descriptor = nullptr;

    hipdnnBackendDescriptor_t _variant_pack = nullptr;

    void SetUp() override
    {
        const std::array<const char*, 1> paths = {"../test_plugins/"};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        destroy_test_handle();
        destroy_test_descriptor(_plan);
        destroy_test_descriptor(_engine_config);
        destroy_test_descriptor(_engine);
        destroy_test_descriptor(_graph_descriptor);
        destroy_test_descriptor(_variant_pack);
    }

    static void destroy_test_descriptor(hipdnnBackendDescriptor_t descriptor)
    {
        if(descriptor != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(descriptor), HIPDNN_STATUS_SUCCESS);
            descriptor = nullptr;
        }
    }

private:
    void destroy_test_handle()
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }
};

TEST_F(Execution_backend_end_api_tests, TestBackendExecuteWithNullHandle)
{
    auto batchnorm_builder = test_util::create_and_populate_batchnorm_node();
    auto serialized_graph = batchnorm_builder.Release();

    test_util::create_and_initialize_backend_descriptor(
        &_graph_descriptor, serialized_graph, _handle);
    test_util::create_test_engine(&_engine, &_graph_descriptor, _handle, GIDX);
    test_util::create_test_engine_config(
        &_engine_config, &_engine, &_graph_descriptor, _handle, GIDX, true);

    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
              HIPDNN_STATUS_SUCCESS);

    test_util::populate_test_execution_plan(
        &_plan, &_engine_config, &_engine, &_graph_descriptor, _handle, GIDX, true);

    ASSERT_EQ(hipdnnBackendExecute(nullptr, _plan, _variant_pack),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(Execution_backend_end_api_tests, TestBackendExecuteWithNullDescriptors)
{
    auto batchnorm_builder = test_util::create_and_populate_batchnorm_node();
    auto serialized_graph = batchnorm_builder.Release();

    test_util::create_and_initialize_backend_descriptor(
        &_graph_descriptor, serialized_graph, _handle);
    test_util::create_test_engine(&_engine, &_graph_descriptor, _handle, GIDX);
    test_util::create_test_engine_config(
        &_engine_config, &_engine, &_graph_descriptor, _handle, GIDX, true);

    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
              HIPDNN_STATUS_SUCCESS);

    test_util::populate_test_execution_plan(
        &_plan, &_engine_config, &_engine, &_graph_descriptor, _handle, GIDX, true);

    ASSERT_EQ(hipdnnBackendExecute(_handle, nullptr, _variant_pack),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, nullptr), HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(Execution_backend_end_api_tests, TestBackendExecuteWithUnfinalizedPlan)
{
    hipdnnBackendDescriptor_t unfinalized_plan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &unfinalized_plan),
        HIPDNN_STATUS_SUCCESS);

    auto batchnorm_builder = test_util::create_and_populate_batchnorm_node();
    auto serialized_graph = batchnorm_builder.Release();

    test_util::create_and_initialize_backend_descriptor(
        &_graph_descriptor, serialized_graph, _handle);

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_variant_pack),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendExecute(_handle, unfinalized_plan, _variant_pack),
              HIPDNN_STATUS_BAD_PARAM);

    destroy_test_descriptor(unfinalized_plan);
}

TEST_F(Execution_backend_end_api_tests, TestBackendExecuteWithWrongDescriptorTypes)
{
    auto batchnorm_builder = test_util::create_and_populate_batchnorm_node();
    auto serialized_graph = batchnorm_builder.Release();

    test_util::create_and_initialize_backend_descriptor(
        &_graph_descriptor, serialized_graph, _handle);
    test_util::create_test_engine(&_engine, &_graph_descriptor, _handle, GIDX);

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_variant_pack),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendExecute(_handle, _engine, _variant_pack), HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
              HIPDNN_STATUS_SUCCESS);

    test_util::populate_test_execution_plan(
        &_plan, &_engine_config, &_engine, &_graph_descriptor, _handle, GIDX, true);
    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, _graph_descriptor), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Execution_backend_end_api_tests, TestBackendExecute)
{
    auto batchnorm_builder = test_util::create_and_populate_batchnorm_node();
    auto serialized_graph = batchnorm_builder.Release();

    std::unordered_map<int64_t, std::string> uid_to_name_map;
    std::unordered_map<std::string, int64_t> name_to_uid_map;
    std::unordered_map<int64_t, std::vector<int64_t>> uid_to_dims_map;

    test_util::extract_tensor_info_from_graph(
        serialized_graph, uid_to_name_map, name_to_uid_map, uid_to_dims_map);

    test_util::create_and_initialize_backend_descriptor(
        &_graph_descriptor, serialized_graph, _handle);
    test_util::create_test_engine(&_engine, &_graph_descriptor, _handle, GIDX);
    test_util::create_test_engine_config(
        &_engine_config, &_engine, &_graph_descriptor, _handle, GIDX, true);

    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
              HIPDNN_STATUS_SUCCESS);

    test_util::populate_test_execution_plan(
        &_plan, &_engine_config, &_engine, &_graph_descriptor, _handle, GIDX, true);

    std::unordered_map<int64_t, void*> data_ptr_mappings;

    for(const auto& [uid, dims] : uid_to_dims_map)
    {
        void* tensor_data
            = test_util::allocate_tensor_memory(dims.data(), dims.size(), HIPDNN_TYPE_FLOAT, true);

        ASSERT_NE(tensor_data, nullptr)
            << "Failed to allocate memory for tensor " << uid_to_name_map[uid];

        data_ptr_mappings[uid] = tensor_data;
    }

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_variant_pack),
              HIPDNN_STATUS_SUCCESS);

    test_util::populate_variant_pack_with_mappings(_variant_pack, data_ptr_mappings, nullptr);

    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, _variant_pack), HIPDNN_STATUS_SUCCESS);

    for(const auto& [uid, data_ptr] : data_ptr_mappings)
    {
        test_util::free_tensor_memory(data_ptr);
    }
}
