// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_frontend/attributes/tensor_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::reference_test_utilities;

struct Bn_2d_test_case
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;

    friend std::ostream& operator<<(std::ostream& ss, const Bn_2d_test_case& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w << ")";
    }

    std::vector<int64_t> get_dims() const
    {
        return {n, c, h, w};
    }
};

template <typename Input_type, typename Intermediate_type>
struct Batchnorm_2d_tensor_bundle
{
    Batchnorm_2d_tensor_bundle(const std::vector<int64_t>& dims, unsigned int seed = 1)
        : derived_dims({1, dims[1], 1, 1})
        , x_tensor(Tensor::make_nchw_tensor<Input_type>(dims))
        , y_tensor(Tensor::make_nchw_tensor<Input_type>(dims))
        , scale_tensor(Tensor::make_nchw_tensor<Intermediate_type>(derived_dims))
        , bias_tensor(Tensor::make_nchw_tensor<Intermediate_type>(derived_dims))
        , mean_tensor(Tensor::make_nchw_tensor<Intermediate_type>(derived_dims))
        , variance_tensor(Tensor::make_nchw_tensor<Intermediate_type>(derived_dims))
    {
        x_tensor.fill_with_random_values<Input_type>(
            static_cast<Input_type>(0.0f), static_cast<Input_type>(1.0f), seed);
        y_tensor.fill_with_random_values<Input_type>(
            static_cast<Input_type>(-100.0f), static_cast<Input_type>(100.0f), seed);

        scale_tensor.fill_with_random_values<Intermediate_type>(
            static_cast<Intermediate_type>(0.0f), static_cast<Intermediate_type>(1.0f), seed);

        bias_tensor.fill_with_random_values<Intermediate_type>(
            static_cast<Intermediate_type>(0.0f), static_cast<Intermediate_type>(1.0f), seed);

        mean_tensor.fill_with_random_values<Intermediate_type>(
            static_cast<Intermediate_type>(0.0f), static_cast<Intermediate_type>(1.0f), seed);

        variance_tensor.fill_with_random_values<Intermediate_type>(
            static_cast<Intermediate_type>(0.1f), static_cast<Intermediate_type>(1.0f), seed);
    }

    std::vector<int64_t> derived_dims;
    Tensor x_tensor;
    Tensor y_tensor;
    Tensor scale_tensor;
    Tensor bias_tensor;
    Tensor mean_tensor;
    Tensor variance_tensor;
};
class Batchnorm_forward_inference_integration_test
    : public ::testing::TestWithParam<Bn_2d_test_case>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Uncomment if you want debug logging info.
        // setenv("HIPDNN_LOG_LEVEL", "info", 1);

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_device_id), hipSuccess);

        //Note: The plugin paths has to be set before we create the hipdnn handle.
        const std::array<const char*, 1> paths = {PLUGIN_DIR};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        // Create handle
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

        //todo: bring back stream support once MigratableMemory supports it
        //ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        //ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    template <typename Input_type, typename Intermediate_type>
    std::unordered_map<int64_t, void*> create_variant_pack(
        const Tensor_attributes& x_tensor_attr,
        const Tensor_attributes& y_tensor_attr,
        const Tensor_attributes& mean_tensor_attr,
        const Tensor_attributes& inv_variance_tensor_attr,
        const Tensor_attributes& scale_tensor_attr,
        const Tensor_attributes& bias_tensor_attr,
        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type>& tensor_bundle)
    {
        std::unordered_map<int64_t, void*> variant_pack;
        variant_pack[x_tensor_attr.get_uid()]
            = tensor_bundle.x_tensor.memory().template device_data<void>();
        variant_pack[mean_tensor_attr.get_uid()]
            = tensor_bundle.mean_tensor.memory().template device_data<void>();
        variant_pack[inv_variance_tensor_attr.get_uid()]
            = tensor_bundle.variance_tensor.memory().template device_data<void>();
        variant_pack[scale_tensor_attr.get_uid()]
            = tensor_bundle.scale_tensor.memory().template device_data<void>();
        variant_pack[bias_tensor_attr.get_uid()]
            = tensor_bundle.bias_tensor.memory().template device_data<void>();
        variant_pack[y_tensor_attr.get_uid()]
            = tensor_bundle.y_tensor.memory().template device_data<void>();

        return variant_pack;
    }

    template <typename Input_type, typename Intermediate_type>
    void run_miopen_batchnorm_fwd(
        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type>& graph_tensor_bundle,
        DataType_t input_data_type,
        DataType_t intermediate_data_type)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();

        graph->set_name("BatchnormInferenceTest");

        int64_t uid = 1;
        auto x_tensor_attr = std::make_shared<Tensor_attributes>();
        x_tensor_attr->set_uid(uid++)
            .set_name("X")
            .set_data_type(input_data_type)
            .set_dim(graph_tensor_bundle.x_tensor.dims())
            .set_stride(graph_tensor_bundle.x_tensor.strides());

        auto mean_tensor_attr = std::make_shared<Tensor_attributes>();
        mean_tensor_attr->set_uid(uid++)
            .set_name("mean")
            .set_data_type(intermediate_data_type)
            .set_dim(graph_tensor_bundle.mean_tensor.dims())
            .set_stride(graph_tensor_bundle.mean_tensor.strides());

        auto inv_variance_tensor_attr = std::make_shared<Tensor_attributes>();
        inv_variance_tensor_attr->set_uid(uid++)
            .set_name("inv_variance")
            .set_data_type(intermediate_data_type)
            .set_dim(graph_tensor_bundle.variance_tensor.dims())
            .set_stride(graph_tensor_bundle.variance_tensor.strides());

        auto scale_tensor_attr = std::make_shared<Tensor_attributes>();
        scale_tensor_attr->set_uid(uid++)
            .set_name("scale")
            .set_data_type(intermediate_data_type)
            .set_dim(graph_tensor_bundle.scale_tensor.dims())
            .set_stride(graph_tensor_bundle.scale_tensor.strides());

        auto bias_tensor_attr = std::make_shared<Tensor_attributes>();
        bias_tensor_attr->set_uid(uid++)
            .set_name("bias")
            .set_data_type(intermediate_data_type)
            .set_dim(graph_tensor_bundle.bias_tensor.dims())
            .set_stride(graph_tensor_bundle.bias_tensor.strides());

        Batchnorm_inference_attributes bn_attrs;
        bn_attrs.set_name("batchnorm_inference");

        auto y_tensor_attr = graph->batchnorm_inference(x_tensor_attr,
                                                        mean_tensor_attr,
                                                        inv_variance_tensor_attr,
                                                        scale_tensor_attr,
                                                        bias_tensor_attr,
                                                        bn_attrs);

        if(!y_tensor_attr->has_uid())
        {
            HIPDNN_LOG_INFO("y_tensor_attr does not have a UID, giving it a UID");
            y_tensor_attr->set_uid(uid++);
        }

        y_tensor_attr->set_data_type(input_data_type);

        // Validate and build graph
        auto result = graph->validate();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->create_execution_plans(_handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        auto variant_pack
            = create_variant_pack<Input_type, Intermediate_type>(*x_tensor_attr,
                                                                 *y_tensor_attr,
                                                                 *mean_tensor_attr,
                                                                 *inv_variance_tensor_attr,
                                                                 *scale_tensor_attr,
                                                                 *bias_tensor_attr,
                                                                 graph_tensor_bundle);

        result = graph->execute(_handle, variant_pack, nullptr);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    }

    template <typename Input_type, typename Intermediate_type>
    void run_cpu_batchnorm_fwd(
        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type>& cpu_tensor_bundle)
    {
        Cpu_fp_reference_implementation<Input_type, Intermediate_type, Intermediate_type>
            cpu_ref_impl;
        cpu_ref_impl.batchnorm_fwd_inference(cpu_tensor_bundle.x_tensor,
                                             cpu_tensor_bundle.scale_tensor,
                                             cpu_tensor_bundle.bias_tensor,
                                             cpu_tensor_bundle.mean_tensor,
                                             cpu_tensor_bundle.variance_tensor,
                                             cpu_tensor_bundle.y_tensor,
                                             1e-3);
    }

    template <typename Input_type, typename Intermediate_type>
    void run_batchnorm_test(const Bn_2d_test_case& test_case, Input_type tolerance = 1e-4f)
    {
        auto input_data_type = get_data_type_enum_from_type<Input_type>();
        auto intermediate_data_type = get_data_type_enum_from_type<Intermediate_type>();

        unsigned int seed = std::random_device{}();
        //log the random seed in case we need to reproduce the test
        HIPDNN_LOG_INFO("Test is using {} for its random seed", seed);

        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type> graph_tensor_bundle(
            test_case.get_dims(), seed);

        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type> cpu_tensor_bundle(
            test_case.get_dims(), seed);

        run_miopen_batchnorm_fwd<Input_type, Intermediate_type>(
            graph_tensor_bundle, input_data_type, intermediate_data_type);
        graph_tensor_bundle.y_tensor.memory().mark_device_modified();

        run_cpu_batchnorm_fwd<Input_type, Intermediate_type>(cpu_tensor_bundle);

        Cpu_fp_reference_validation<Input_type> cpu_ref_validation(tolerance, tolerance);
        EXPECT_TRUE(cpu_ref_validation.compare_buffers(cpu_tensor_bundle.y_tensor.memory(),
                                                       graph_tensor_bundle.y_tensor.memory()));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _device_id = 0;
};

std::vector<Bn_2d_test_case> get_bn_fwd_inference_test_cases()
{
    return {
        {.n = 1, .c = 3, .h = 14, .w = 14},
        {.n = 2, .c = 3, .h = 14, .w = 14},
        {.n = 64, .c = 3, .h = 14, .w = 14},
        {.n = 64, .c = 256, .h = 14, .w = 14},
        {.n = 64, .c = 256, .h = 28, .w = 28},
        {.n = 64, .c = 256, .h = 56, .w = 56},
        {.n = 64, .c = 512, .h = 14, .w = 14},
        {.n = 64, .c = 512, .h = 28, .w = 28},
        {.n = 64, .c = 512, .h = 7, .w = 7},
        {.n = 64, .c = 64, .h = 112, .w = 112},
        {.n = 64, .c = 64, .h = 56, .w = 56},
    };
}

TEST_P(Batchnorm_forward_inference_integration_test, RunFloatFwdBatchnormGraph)
{
    Bn_2d_test_case test_case = GetParam();
    run_batchnorm_test<float, float>(test_case, 1e-6f);
}

INSTANTIATE_TEST_SUITE_P(RunFloatFwdBatchnormGraph,
                         Batchnorm_forward_inference_integration_test,
                         testing::ValuesIn(get_bn_fwd_inference_test_cases()));

class Batchnorm_forward_inference_integration_test_bfloat16
    : public Batchnorm_forward_inference_integration_test
{
};

TEST_P(Batchnorm_forward_inference_integration_test_bfloat16, RunBfloat16FwdBatchnormGraph)
{
    Bn_2d_test_case test_case = GetParam();
    run_batchnorm_test<hip_bfloat16, float>(test_case, 1e-2_bf);
}

INSTANTIATE_TEST_SUITE_P(RunBfloat16FwdBatchnormGraph,
                         Batchnorm_forward_inference_integration_test_bfloat16,
                         testing::ValuesIn(get_bn_fwd_inference_test_cases()));

class Batchnorm_forward_inference_integration_test_half
    : public Batchnorm_forward_inference_integration_test
{
};
TEST_P(Batchnorm_forward_inference_integration_test_half, RunHalfFwdbatchnormGraph)
{
    Bn_2d_test_case test_case = GetParam();
    run_batchnorm_test<half, float>(test_case, 1e-2_h);
}

INSTANTIATE_TEST_SUITE_P(RunHalfFwdbatchnormGraph,
                         Batchnorm_forward_inference_integration_test_half,
                         testing::ValuesIn(get_bn_fwd_inference_test_cases()));
