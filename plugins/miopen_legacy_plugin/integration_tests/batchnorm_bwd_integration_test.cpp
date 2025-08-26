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
#include <hipdnn_frontend/utilities.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::reference_test_utilities;

namespace
{

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
    Batchnorm_2d_tensor_bundle(const std::vector<int64_t>& dims,
                               unsigned int seed = 1,
                               const Tensor_layout& layout = Tensor_layout::NCHW)
        : derived_dims({1, dims[1], 1, 1})
        , x_tensor(dims, layout)
        , dy_tensor(dims, layout)
        , dx_tensor(dims, layout)
        , scale_tensor(derived_dims)
        , dscale_tensor(derived_dims)
        , dbias_tensor(derived_dims)
        , mean_tensor(derived_dims)
        , inv_variance_tensor(derived_dims)
    {
        x_tensor.fill_with_random_values(
            static_cast<Input_type>(-1.0f), static_cast<Input_type>(1.0f), seed);

        // Keep dy & scale as low values, since they become huge with large size tensors, and blow up precision.
        dy_tensor.fill_with_random_values(
            static_cast<Input_type>(-0.1f), static_cast<Input_type>(0.1f), seed);
        scale_tensor.fill_with_random_values(
            static_cast<Intermediate_type>(-0.1f), static_cast<Intermediate_type>(0.1f), seed);

        // Mean assuming a large # of samples will trend towards mid point of x range
        // Setting to 0.
        mean_tensor.fill_with_random_values(
            static_cast<Intermediate_type>(-0.1f), static_cast<Intermediate_type>(0.1f), seed);

        // inv_variance is 1/sqrt(variance + epsilon) and needs to be positive in all cases
        // Based off X & mean calc, we are setting it close to 2.0f in this case
        inv_variance_tensor.fill_with_random_values(
            static_cast<Intermediate_type>(1.9f), static_cast<Intermediate_type>(2.0f), seed);
    }

    std::vector<int64_t> derived_dims;
    PinnedTensor<Input_type> x_tensor;
    PinnedTensor<Input_type> dy_tensor;
    PinnedTensor<Input_type> dx_tensor;
    PinnedTensor<Intermediate_type> scale_tensor;
    PinnedTensor<Intermediate_type> dscale_tensor;
    PinnedTensor<Intermediate_type> dbias_tensor;
    PinnedTensor<Intermediate_type> mean_tensor;
    PinnedTensor<Intermediate_type> inv_variance_tensor;
};

} // namespace

class Batchnorm_backward_integration_test : public ::testing::TestWithParam<Bn_2d_test_case>
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
        const Tensor_attributes& dy_tensor_attr,
        const Tensor_attributes& dx_tensor_attr,
        const Tensor_attributes& scale_tensor_attr,
        const Tensor_attributes& dscale_tensor_attr,
        const Tensor_attributes& dbias_tensor_attr,
        const Tensor_attributes& mean_tensor_attr,
        const Tensor_attributes& inv_variance_tensor_attr,
        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type>& tensor_bundle)
    {
        std::unordered_map<int64_t, void*> variant_pack;
        variant_pack[x_tensor_attr.get_uid()] = tensor_bundle.x_tensor.memory().device_data();
        variant_pack[dy_tensor_attr.get_uid()] = tensor_bundle.dy_tensor.memory().device_data();
        variant_pack[dx_tensor_attr.get_uid()] = tensor_bundle.dx_tensor.memory().device_data();
        variant_pack[scale_tensor_attr.get_uid()]
            = tensor_bundle.scale_tensor.memory().device_data();
        variant_pack[dscale_tensor_attr.get_uid()]
            = tensor_bundle.dscale_tensor.memory().device_data();
        variant_pack[dbias_tensor_attr.get_uid()]
            = tensor_bundle.dbias_tensor.memory().device_data();
        variant_pack[mean_tensor_attr.get_uid()] = tensor_bundle.mean_tensor.memory().device_data();
        variant_pack[inv_variance_tensor_attr.get_uid()]
            = tensor_bundle.inv_variance_tensor.memory().device_data();

        return variant_pack;
    }

    template <typename Input_type, typename Intermediate_type>
    void run_miopen_batchnorm_bwd(
        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type>& graph_tensor_bundle,
        DataType_t input_data_type,
        DataType_t intermediate_data_type)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();

        graph->set_name("BatchnormBackwardTest");

        int64_t uid = 1;

        auto x_attr = make_tensor_attributes("x", input_data_type, graph_tensor_bundle.x_tensor);
        x_attr.set_uid(uid++);
        auto x_tensor_attr = std::make_shared<Tensor_attributes>(std::move(x_attr));

        auto dy_attr = make_tensor_attributes("dy", input_data_type, graph_tensor_bundle.dy_tensor);
        dy_attr.set_uid(uid++);
        auto dy_tensor_attr = std::make_shared<Tensor_attributes>(std::move(dy_attr));

        auto scale_attr = make_tensor_attributes(
            "scale", intermediate_data_type, graph_tensor_bundle.scale_tensor);
        scale_attr.set_uid(uid++);
        auto scale_tensor_attr = std::make_shared<Tensor_attributes>(std::move(scale_attr));

        auto mean_attr = make_tensor_attributes(
            "mean", intermediate_data_type, graph_tensor_bundle.mean_tensor);
        mean_attr.set_uid(uid++);
        auto mean_tensor_attr = std::make_shared<Tensor_attributes>(std::move(mean_attr));

        auto inv_variance_attr = make_tensor_attributes(
            "inv_variance", intermediate_data_type, graph_tensor_bundle.inv_variance_tensor);
        inv_variance_attr.set_uid(uid++);
        auto inv_variance_tensor_attr
            = std::make_shared<Tensor_attributes>(std::move(inv_variance_attr));

        Batchnorm_backward_attributes bn_attrs;
        bn_attrs.set_name("batchnorm_backward");
        bn_attrs.set_saved_mean_and_inv_variance(mean_tensor_attr, inv_variance_tensor_attr);

        auto output_tensors_attr
            = graph->batchnorm_backward(dy_tensor_attr, x_tensor_attr, scale_tensor_attr, bn_attrs);

        auto& dx_tensor_attr = output_tensors_attr[0];
        if(!dx_tensor_attr->has_uid())
        {
            HIPDNN_LOG_INFO("dx_tensor_attr does not have a UID, giving it a UID");
            dx_tensor_attr->set_uid(uid++);
        }
        dx_tensor_attr->set_data_type(input_data_type);

        auto& dscale_tensor_attr = output_tensors_attr[1];
        if(!dscale_tensor_attr->has_uid())
        {
            HIPDNN_LOG_INFO("dscale_tensor_attr does not have a UID, giving it a UID");
            dscale_tensor_attr->set_uid(uid++);
        }
        dscale_tensor_attr->set_data_type(intermediate_data_type);

        auto& dbias_tensor_attr = output_tensors_attr[2];
        if(!dbias_tensor_attr->has_uid())
        {
            HIPDNN_LOG_INFO("dbias_tensor_attr does not have a UID, giving it a UID");
            dbias_tensor_attr->set_uid(uid++);
        }
        dbias_tensor_attr->set_data_type(intermediate_data_type);

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
                                                                 *dy_tensor_attr,
                                                                 *dx_tensor_attr,
                                                                 *scale_tensor_attr,
                                                                 *dscale_tensor_attr,
                                                                 *dbias_tensor_attr,
                                                                 *mean_tensor_attr,
                                                                 *inv_variance_tensor_attr,
                                                                 graph_tensor_bundle);

        result = graph->execute(_handle, variant_pack, nullptr);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    }

    template <typename Input_type, typename Intermediate_type>
    void run_cpu_batchnorm_bwd(
        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type>& cpu_tensor_bundle)
    {
        Cpu_fp_reference_implementation<Input_type, Intermediate_type, Intermediate_type>
            cpu_ref_impl;
        cpu_ref_impl.batchnorm_bwd(cpu_tensor_bundle.dy_tensor,
                                   cpu_tensor_bundle.x_tensor,
                                   cpu_tensor_bundle.mean_tensor,
                                   cpu_tensor_bundle.inv_variance_tensor,
                                   cpu_tensor_bundle.scale_tensor,
                                   cpu_tensor_bundle.dx_tensor,
                                   cpu_tensor_bundle.dscale_tensor,
                                   cpu_tensor_bundle.dbias_tensor);
    }

    template <typename Input_type, typename Intermediate_type>
    void run_batchnorm_test(const Bn_2d_test_case& test_case,
                            Input_type tolerance = 1e4f,
                            const Tensor_layout& layout = Tensor_layout::NCHW)
    {
        auto input_data_type = get_data_type_enum_from_type<Input_type>();
        auto intermediate_data_type = get_data_type_enum_from_type<Intermediate_type>();

        unsigned int seed = std::random_device{}();
        //log the random seed in case we need to reproduce the test
        HIPDNN_LOG_INFO("Test is using {} for its random seed", seed);

        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type> graph_tensor_bundle(
            test_case.get_dims(), seed, layout);

        Batchnorm_2d_tensor_bundle<Input_type, Intermediate_type> cpu_tensor_bundle(
            test_case.get_dims(), seed, layout);

        run_miopen_batchnorm_bwd<Input_type, Intermediate_type>(
            graph_tensor_bundle, input_data_type, intermediate_data_type);
        graph_tensor_bundle.dx_tensor.memory().mark_device_modified();
        graph_tensor_bundle.dscale_tensor.memory().mark_device_modified();
        graph_tensor_bundle.dbias_tensor.memory().mark_device_modified();

        run_cpu_batchnorm_bwd<Input_type, Intermediate_type>(cpu_tensor_bundle);

        Cpu_fp_reference_validation<Input_type> cpu_ref_validation(tolerance, tolerance);
        EXPECT_TRUE(cpu_ref_validation.all_close(cpu_tensor_bundle.dx_tensor.memory(),
                                                 graph_tensor_bundle.dx_tensor.memory()));

        Cpu_fp_reference_validation<Intermediate_type> cpu_ref_intermediate_validation(tolerance,
                                                                                       tolerance);
        EXPECT_TRUE(cpu_ref_intermediate_validation.all_close(
            cpu_tensor_bundle.dscale_tensor.memory(), graph_tensor_bundle.dscale_tensor.memory()));
        EXPECT_TRUE(cpu_ref_intermediate_validation.all_close(
            cpu_tensor_bundle.dbias_tensor.memory(), graph_tensor_bundle.dbias_tensor.memory()));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _device_id = 0;
};

class Batchnorm_backward_integration_test_bfloat16 : public Batchnorm_backward_integration_test
{
};

class Batchnorm_backward_integration_test_half : public Batchnorm_backward_integration_test
{
};

class Batchnorm_backward_integration_test_nhwc : public Batchnorm_backward_integration_test
{
};

namespace
{

std::vector<Bn_2d_test_case> get_bn_bwd_test_cases()
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

} // namespace

// Note:
// Tolerance ranges are set to be 4e-3f due to batchnorm being numerical unstable for large tensor sizes.
// MIOpen uses 4e-3f for it's batchnorm tests to verify, but it uses RMS calc instead of all_close type check.
// You can swap the tests above to use cpu_fp_reference_miopen_rms_validation if you want to match MIOpen's tolerance checks exactly.
TEST_P(Batchnorm_backward_integration_test, RunFloatBwdBatchnormGraphNCHW)
{
    Bn_2d_test_case test_case = GetParam();
    run_batchnorm_test<float, float>(test_case, 4e-3f);
}

INSTANTIATE_TEST_SUITE_P(RunFloatBwdBatchnormGraph,
                         Batchnorm_backward_integration_test,
                         testing::ValuesIn(get_bn_bwd_test_cases()));

TEST_P(Batchnorm_backward_integration_test_bfloat16, RunBfloat16BwdBatchnormGraphNCHW)
{
    Bn_2d_test_case test_case = GetParam();
    run_batchnorm_test<hip_bfloat16, float>(test_case, 4e-3_bf);
}

INSTANTIATE_TEST_SUITE_P(RunBfloat16BwdBatchnormGraph,
                         Batchnorm_backward_integration_test_bfloat16,
                         testing::ValuesIn(get_bn_bwd_test_cases()));

TEST_P(Batchnorm_backward_integration_test_half, RunHalfBwdBatchnormGraphNCWH)
{
    Bn_2d_test_case test_case = GetParam();
    run_batchnorm_test<half, float>(test_case, 4e-3_h);
}

INSTANTIATE_TEST_SUITE_P(RunHalfBwdBatchnormGraph,
                         Batchnorm_backward_integration_test_half,
                         testing::ValuesIn(get_bn_bwd_test_cases()));

TEST_P(Batchnorm_backward_integration_test_nhwc, RunFloatBwdBatchnormGraphNHWC)
{
    Bn_2d_test_case test_case = GetParam();
    run_batchnorm_test<float, float>(test_case, 4e-3f, Tensor_layout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(RunFloatBwdBatchnormGraphNHWC,
                         Batchnorm_backward_integration_test_nhwc,
                         testing::ValuesIn(get_bn_bwd_test_cases()));
