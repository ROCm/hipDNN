// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_frontend/attributes/tensor_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_frontend/utilities.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>
#include <test_plugins/test_plugin_constants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_sdk::utilities;

enum class FailurePoint
{
    NONE, // No failure expected
    CREATE_EXECUTION_PLAN, // Expect failure at create execution plan
    EXECUTE // Expect failure at execute
};

struct Integration_test_case
{
    std::string plugin_path;
    std::string description;
    std::string graph_name;
    FailurePoint expected_failure;
    bool use_manual_uids;

    friend std::ostream& operator<<(std::ostream& os, const Integration_test_case& tc)
    {
        os << "BatchnormTestCase{"
           << "plugin_path: " << tc.plugin_path << ", description: " << tc.description
           << ", graph_name: " << tc.graph_name << ", expected_failure: ";

        switch(tc.expected_failure)
        {
        case FailurePoint::NONE:
            os << "NONE";
            break;
        case FailurePoint::CREATE_EXECUTION_PLAN:
            os << "CREATE_EXECUTION_PLAN";
            break;
        case FailurePoint::EXECUTE:
            os << "EXECUTE";
            break;
        default:
            os << "UNKNOWN";
            break;
        }

        os << ", use_manual_uids: " << (tc.use_manual_uids ? "true" : "false") << "}";

        return os;
    }
};

// Test class for frontend integration tests.
// This class runs end-to-end tests to ensure the frontend API works as expected.
// Notes:
//        - We are using fake test plugins to simulate different scenarios.
//        - The tests will validate the graph creation, execution plan building, and execution through the full flow.
//        - We are using a batchnorm graph since the graph doesn't really matter due to fake plugins.
class Frontend_e2e_integration_test : public ::testing::TestWithParam<Integration_test_case>
{
protected:
    // Simplified tensor bundle for frontend integration tests
    template <typename Input_type, typename Intermediate_type>
    struct Simple_batchnorm_2d_tensor_bundle
    {
        Simple_batchnorm_2d_tensor_bundle(const std::vector<int64_t>& dims)
            : derived_dims({1, dims[1], 1, 1})
            , x_tensor(Tensor::make_nchw_tensor<Input_type>(dims))
            , y_tensor(Tensor::make_nchw_tensor<Input_type>(dims))
            , scale_tensor(Tensor::make_nchw_tensor<Intermediate_type>(derived_dims))
            , bias_tensor(Tensor::make_nchw_tensor<Intermediate_type>(derived_dims))
            , mean_tensor(Tensor::make_nchw_tensor<Intermediate_type>(derived_dims))
            , variance_tensor(Tensor::make_nchw_tensor<Intermediate_type>(derived_dims))
        {
            // Initialize with simple constant values
            x_tensor.fill_with_value<Input_type>(static_cast<Input_type>(1.0f));
            y_tensor.fill_with_value<Input_type>(static_cast<Input_type>(0.0f));
            scale_tensor.fill_with_value<Intermediate_type>(static_cast<Intermediate_type>(1.0f));
            bias_tensor.fill_with_value<Intermediate_type>(static_cast<Intermediate_type>(0.0f));
            mean_tensor.fill_with_value<Intermediate_type>(static_cast<Intermediate_type>(0.0f));
            variance_tensor.fill_with_value<Intermediate_type>(
                static_cast<Intermediate_type>(1.0f));
        }

        std::vector<int64_t> derived_dims;
        Tensor x_tensor;
        Tensor y_tensor;
        Tensor scale_tensor;
        Tensor bias_tensor;
        Tensor mean_tensor;
        Tensor variance_tensor;
    };

    struct Batchnorm_test_tensors
    {
        std::shared_ptr<Tensor_attributes> x;
        std::shared_ptr<Tensor_attributes> mean;
        std::shared_ptr<Tensor_attributes> inv_variance;
        std::shared_ptr<Tensor_attributes> scale;
        std::shared_ptr<Tensor_attributes> bias;
        std::shared_ptr<Tensor_attributes> y;
    };

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    static hipdnnHandle_t setup_test_environment_with_plugin(const std::string& plugin_path)
    {
        EXPECT_EQ(hipInit(0), hipSuccess);
        int device_id = 0;
        EXPECT_EQ(hipGetDevice(&device_id), hipSuccess);

        // Set up plugin path - load specific plugin by absolute path
        const std::array<const char*, 1> paths = {plugin_path.c_str()};
        EXPECT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        // Create handle
        hipdnnHandle_t handle = nullptr;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        return handle;
    }

    static std::pair<std::shared_ptr<Graph>, Batchnorm_test_tensors>
        create_batchnorm_test_graph_with_uids(
            const std::string& graph_name,
            const Simple_batchnorm_2d_tensor_bundle<float, float>& tensor_bundle,
            bool use_manual_uids)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
        graph->set_name(graph_name);

        int64_t uid = 1;
        Batchnorm_test_tensors tensors;

        auto x_attr = make_tensor_attributes("X", DataType_t::FLOAT, tensor_bundle.x_tensor);
        if(use_manual_uids)
        {
            x_attr.set_uid(uid++);
        }
        tensors.x = std::make_shared<Tensor_attributes>(std::move(x_attr));

        auto mean_attr
            = make_tensor_attributes("mean", DataType_t::FLOAT, tensor_bundle.mean_tensor);
        if(use_manual_uids)
        {
            mean_attr.set_uid(uid++);
        }
        tensors.mean = std::make_shared<Tensor_attributes>(std::move(mean_attr));

        auto inv_variance_attr = make_tensor_attributes(
            "inv_variance", DataType_t::FLOAT, tensor_bundle.variance_tensor);
        if(use_manual_uids)
        {
            inv_variance_attr.set_uid(uid++);
        }
        tensors.inv_variance = std::make_shared<Tensor_attributes>(std::move(inv_variance_attr));

        auto scale_attr
            = make_tensor_attributes("scale", DataType_t::FLOAT, tensor_bundle.scale_tensor);
        if(use_manual_uids)
        {
            scale_attr.set_uid(uid++);
        }
        tensors.scale = std::make_shared<Tensor_attributes>(std::move(scale_attr));

        auto bias_attr
            = make_tensor_attributes("bias", DataType_t::FLOAT, tensor_bundle.bias_tensor);
        if(use_manual_uids)
        {
            bias_attr.set_uid(uid++);
        }
        tensors.bias = std::make_shared<Tensor_attributes>(std::move(bias_attr));

        Batchnorm_inference_attributes bn_attrs;
        bn_attrs.set_name("batchnorm_inference");

        tensors.y = graph->batchnorm_inference(
            tensors.x, tensors.mean, tensors.inv_variance, tensors.scale, tensors.bias, bn_attrs);

        if(use_manual_uids)
        {
            tensors.y->set_uid(uid++);
        }
        tensors.y->set_data_type(DataType_t::FLOAT);

        return {graph, tensors};
    }

    static std::unordered_map<int64_t, void*>
        create_variant_pack(const Batchnorm_test_tensors& tensors,
                            const Simple_batchnorm_2d_tensor_bundle<float, float>& tensor_bundle)
    {
        std::unordered_map<int64_t, void*> variant_pack;
        variant_pack[tensors.x->get_uid()]
            = const_cast<void*>(tensor_bundle.x_tensor.memory().template device_data<void>());
        variant_pack[tensors.mean->get_uid()]
            = const_cast<void*>(tensor_bundle.mean_tensor.memory().template device_data<void>());
        variant_pack[tensors.inv_variance->get_uid()] = const_cast<void*>(
            tensor_bundle.variance_tensor.memory().template device_data<void>());
        variant_pack[tensors.scale->get_uid()]
            = const_cast<void*>(tensor_bundle.scale_tensor.memory().template device_data<void>());
        variant_pack[tensors.bias->get_uid()]
            = const_cast<void*>(tensor_bundle.bias_tensor.memory().template device_data<void>());
        variant_pack[tensors.y->get_uid()]
            = const_cast<void*>(tensor_bundle.y_tensor.memory().template device_data<void>());

        return variant_pack;
    }

    static void
        run_graph_pipeline(const std::shared_ptr<Graph>& graph,
                           hipdnnHandle_t handle,
                           const Batchnorm_test_tensors& tensors,
                           const Simple_batchnorm_2d_tensor_bundle<float, float>& tensor_bundle,
                           FailurePoint expected_failure = FailurePoint::NONE)
    {
        auto result = graph->validate();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->build_operation_graph(handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->create_execution_plans(handle);
        if(expected_failure == FailurePoint::CREATE_EXECUTION_PLAN)
        {
            ASSERT_NE(result.code, error_code_t::OK) << "create_execution_plans should fail";
            return;
        }
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        ASSERT_TRUE(tensors.x->has_uid());
        ASSERT_TRUE(tensors.mean->has_uid());
        ASSERT_TRUE(tensors.inv_variance->has_uid());
        ASSERT_TRUE(tensors.scale->has_uid());
        ASSERT_TRUE(tensors.bias->has_uid());
        ASSERT_TRUE(tensors.y->has_uid());

        auto variant_pack = create_variant_pack(tensors, tensor_bundle);

        result = graph->execute(handle, variant_pack, nullptr);
        if(expected_failure == FailurePoint::EXECUTE)
        {
            ASSERT_NE(result.code, error_code_t::OK) << "Execute should fail";
        }
        else
        {
            ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
        }
    }

    void run_test()
    {
        const auto& test_case = GetParam();

        // Setup environment with specified plugin
        _handle = setup_test_environment_with_plugin(test_case.plugin_path);

        // Create tensor bundle
        std::vector<int64_t> dims = {2, 3, 14, 14}; // n=2, c=3, h=14, w=14
        Simple_batchnorm_2d_tensor_bundle<float, float> tensor_bundle(dims);

        // Create graph and tensors using the unified function
        auto [graph, tensors] = create_batchnorm_test_graph_with_uids(
            test_case.graph_name, tensor_bundle, test_case.use_manual_uids);

        run_graph_pipeline(graph, _handle, tensors, tensor_bundle, test_case.expected_failure);
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

INSTANTIATE_TEST_SUITE_P(
    IntegrationTests,
    Frontend_e2e_integration_test,
    ::testing::Values(Integration_test_case{hipdnn_tests::plugin_constants::test_good_plugin_path(),
                                            "Default plugin with manual UIDs",
                                            "DefaultPluginBatchnormTest",
                                            FailurePoint::NONE,
                                            true},
                      Integration_test_case{hipdnn_tests::plugin_constants::test_good_plugin_path(),
                                            "Default plugin with auto UIDs",
                                            "DefaultPluginBatchnormTestAutoUID",
                                            FailurePoint::NONE,
                                            false},
                      Integration_test_case{
                          hipdnn_tests::plugin_constants::test_execute_fails_plugin_path(),
                          "Execute fails plugin",
                          "ExecuteFailsPluginBatchnormTest",
                          FailurePoint::EXECUTE,
                          true},
                      Integration_test_case{
                          hipdnn_tests::plugin_constants::test_no_applicable_engines_plugin_path(),
                          "No applicable engines plugin",
                          "NoEnginesPluginBatchnormTest",
                          FailurePoint::CREATE_EXECUTION_PLAN,
                          true}),
    // Provide a custom name for each test instance
    [](const ::testing::TestParamInfo<Integration_test_case>& info) {
        std::string name = info.param.description;
        std::ranges::replace_if(name, [](char c) { return !std::isalnum(c); }, '_');
        return name;
    });

TEST_P(Frontend_e2e_integration_test, IntegrationTest)
{
    run_test();
}
