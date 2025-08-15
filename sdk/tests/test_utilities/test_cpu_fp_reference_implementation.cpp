// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/half_utils.hpp>
#include <hipdnn_sdk/utilities/hip_bfloat16_utils.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

using namespace hipdnn_sdk::reference_test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

TEST(CpuFpReferenceImplementation, BatchnormInferFloatUsage)
{
    SKIP_IF_NO_DEVICES();

    auto input_tensor = Tensor::make_tensor<float>({1, 3, 224, 224});
    auto output_tensor = Tensor::make_tensor<float>({1, 3, 224, 224});
    auto bias_tensor = Tensor::make_tensor<float>({1, 3});
    auto scale_tensor = Tensor::make_tensor<float>({1, 3});
    auto mean_tensor = Tensor::make_tensor<float>({1, 3});
    auto variance_tensor = Tensor::make_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<float, float, float> ref_impl;

    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BatchnormInferBFloat16Usage)
{
    SKIP_IF_NO_DEVICES();

    auto input_tensor = Tensor::make_tensor<hip_bfloat16>({1, 3, 224, 224});
    auto output_tensor = Tensor::make_tensor<hip_bfloat16>({1, 3, 224, 224});
    auto bias_tensor = Tensor::make_tensor<float>({1, 3});
    auto scale_tensor = Tensor::make_tensor<float>({1, 3});
    auto mean_tensor = Tensor::make_tensor<float>({1, 3});
    auto variance_tensor = Tensor::make_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<hip_bfloat16, float, float> ref_impl;

    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BatchnormInferHalfUsage)
{
    SKIP_IF_NO_DEVICES();

    auto input_tensor = Tensor::make_tensor<half>({1, 3, 224, 224});
    auto output_tensor = Tensor::make_tensor<half>({1, 3, 224, 224});
    auto bias_tensor = Tensor::make_tensor<float>({1, 3});
    auto scale_tensor = Tensor::make_tensor<float>({1, 3});
    auto mean_tensor = Tensor::make_tensor<float>({1, 3});
    auto variance_tensor = Tensor::make_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<half, float, float> ref_impl;

    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);
}

TEST(CpuFpReferenceImplementaion, BatchnormInferDoubleUsage)
{
    SKIP_IF_NO_DEVICES();

    auto input_tensor = Tensor::make_tensor<double>({1, 3, 224, 224});
    auto output_tensor = Tensor::make_tensor<double>({1, 3, 224, 224});
    auto bias_tensor = Tensor::make_tensor<double>({1, 3});
    auto scale_tensor = Tensor::make_tensor<double>({1, 3});
    auto mean_tensor = Tensor::make_tensor<double>({1, 3});
    auto variance_tensor = Tensor::make_tensor<double>({1, 3});

    Cpu_fp_reference_implementation<double, double, double> ref_impl;

    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BatchnormInferFloatUsageNHWC)
{
    SKIP_IF_NO_DEVICES();

    auto input_tensor = Tensor::make_tensor<float>({6, 3, 32, 32}, Tensor_layout::NHWC);
    auto output_tensor = Tensor::make_tensor<float>({6, 3, 32, 32}, Tensor_layout::NHWC);
    auto bias_tensor = Tensor::make_tensor<float>({1, 3, 1, 1});
    auto scale_tensor = Tensor::make_tensor<float>({1, 3, 1, 1});
    auto mean_tensor = Tensor::make_tensor<float>({1, 3, 1, 1});
    auto variance_tensor = Tensor::make_tensor<float>({1, 3, 1, 1});

    Cpu_fp_reference_implementation<float, float, float> ref_impl;

    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BatchnormInferSanityValidation)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 1, 2, 2};

    auto input_tensor = Tensor::make_tensor<double>(dims);
    auto output_tensor = Tensor::make_tensor<double>(dims);
    auto scale_tensor = Tensor::make_tensor<double>({1, 1, 1, 1});
    auto bias_tensor = Tensor::make_tensor<double>({1, 1, 1, 1});
    auto mean_tensor = Tensor::make_tensor<double>({1, 1, 1, 1});
    auto variance_tensor = Tensor::make_tensor<double>({1, 1, 1, 1});

    // x = [1, 2, 3, 4]
    input_tensor.set_host_value<double>(0, 0, 0, 0, 1.0);
    input_tensor.set_host_value<double>(0, 0, 0, 1, 2.0);
    input_tensor.set_host_value<double>(0, 0, 1, 0, 3.0);
    input_tensor.set_host_value<double>(0, 0, 1, 1, 4.0);

    // fixed scale and bias parameters (one channel)
    scale_tensor.set_host_value<double>(0, 0, 0, 0, 2.0);
    bias_tensor.set_host_value<double>(0, 0, 0, 0, 0.5);

    // inference uses population statistics per channel:
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 5.0 / 4 = 1.25
    // (in practice, computed during training)
    mean_tensor.set_host_value<double>(0, 0, 0, 0, 2.5);
    variance_tensor.set_host_value<double>(0, 0, 0, 0, 1.25);

    // output is calculated via a pointwise linear transform on x:
    // y = scale * (x - mean) * inv_variance + bias = 2 * (x - 2.5) * inv_variance + 0.5
    // where inv_variance (named by convention) = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    const std::vector<double> expected_output = {-2.18327084, -0.39442361, 1.39442361, 3.18327084};

    Cpu_fp_reference_implementation<double, double, double> ref_impl;
    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);

    auto tolerance = 1e-6;

    EXPECT_NEAR(output_tensor.get_host_value<double>(0, 0, 0, 0), expected_output[0], tolerance);
    EXPECT_NEAR(output_tensor.get_host_value<double>(0, 0, 0, 1), expected_output[1], tolerance);
    EXPECT_NEAR(output_tensor.get_host_value<double>(0, 0, 1, 0), expected_output[2], tolerance);
    EXPECT_NEAR(output_tensor.get_host_value<double>(0, 0, 1, 1), expected_output[3], tolerance);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdFloatUsage)
{
    SKIP_IF_NO_DEVICES();

    auto x_tensor = Tensor::make_tensor<float>({6, 3, 32, 32});
    auto dy_tensor = Tensor::make_tensor<float>({6, 3, 32, 32});
    auto dx_tensor = Tensor::make_tensor<float>({6, 3, 32, 32});
    auto scale_tensor = Tensor::make_tensor<float>({1, 3});
    auto mean_tensor = Tensor::make_tensor<float>({1, 3});
    auto inv_variance_tensor = Tensor::make_tensor<float>({1, 3});
    auto dscale_tensor = Tensor::make_tensor<float>({1, 3});
    auto dbias_tensor = Tensor::make_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<float, float, float> ref_impl;

    ref_impl.batchnorm_bwd(dy_tensor,
                           x_tensor,
                           mean_tensor,
                           inv_variance_tensor,
                           scale_tensor,
                           dx_tensor,
                           dscale_tensor,
                           dbias_tensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdBFloat16Usage)
{
    SKIP_IF_NO_DEVICES();

    auto x_tensor = Tensor::make_tensor<hip_bfloat16>({6, 3, 32, 32});
    auto dy_tensor = Tensor::make_tensor<hip_bfloat16>({6, 3, 32, 32});
    auto dx_tensor = Tensor::make_tensor<hip_bfloat16>({6, 3, 32, 32});
    auto scale_tensor = Tensor::make_tensor<float>({1, 3});
    auto mean_tensor = Tensor::make_tensor<float>({1, 3});
    auto inv_variance_tensor = Tensor::make_tensor<float>({1, 3});
    auto dscale_tensor = Tensor::make_tensor<float>({1, 3});
    auto dbias_tensor = Tensor::make_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<hip_bfloat16, float, float> ref_impl;

    ref_impl.batchnorm_bwd(dy_tensor,
                           x_tensor,
                           mean_tensor,
                           inv_variance_tensor,
                           scale_tensor,
                           dx_tensor,
                           dscale_tensor,
                           dbias_tensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdHalfUsage)
{
    SKIP_IF_NO_DEVICES();

    auto x_tensor = Tensor::make_tensor<half>({6, 3, 32, 32});
    auto dy_tensor = Tensor::make_tensor<half>({6, 3, 32, 32});
    auto dx_tensor = Tensor::make_tensor<half>({6, 3, 32, 32});
    auto scale_tensor = Tensor::make_tensor<float>({1, 3});
    auto mean_tensor = Tensor::make_tensor<float>({1, 3});
    auto inv_variance_tensor = Tensor::make_tensor<float>({1, 3});
    auto dscale_tensor = Tensor::make_tensor<float>({1, 3});
    auto dbias_tensor = Tensor::make_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<half, float, float> ref_impl;

    ref_impl.batchnorm_bwd(dy_tensor,
                           x_tensor,
                           mean_tensor,
                           inv_variance_tensor,
                           scale_tensor,
                           dx_tensor,
                           dscale_tensor,
                           dbias_tensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdDoubleUsage)
{
    SKIP_IF_NO_DEVICES();

    auto x_tensor = Tensor::make_tensor<double>({6, 3, 32, 32});
    auto dy_tensor = Tensor::make_tensor<double>({6, 3, 32, 32});
    auto dx_tensor = Tensor::make_tensor<double>({6, 3, 32, 32});
    auto scale_tensor = Tensor::make_tensor<double>({1, 3});
    auto mean_tensor = Tensor::make_tensor<double>({1, 3});
    auto inv_variance_tensor = Tensor::make_tensor<double>({1, 3});
    auto dscale_tensor = Tensor::make_tensor<double>({1, 3});
    auto dbias_tensor = Tensor::make_tensor<double>({1, 3});

    Cpu_fp_reference_implementation<double, double, double> ref_impl;

    ref_impl.batchnorm_bwd(dy_tensor,
                           x_tensor,
                           mean_tensor,
                           inv_variance_tensor,
                           scale_tensor,
                           dx_tensor,
                           dscale_tensor,
                           dbias_tensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdFloatUsageNHWC)
{
    SKIP_IF_NO_DEVICES();

    auto x_tensor = Tensor::make_tensor<float>({6, 3, 32, 32}, Tensor_layout::NHWC);
    auto dy_tensor = Tensor::make_tensor<float>({6, 3, 32, 32}, Tensor_layout::NHWC);
    auto dx_tensor = Tensor::make_tensor<float>({6, 3, 32, 32});
    auto scale_tensor = Tensor::make_tensor<float>({1, 3});
    auto mean_tensor = Tensor::make_tensor<float>({1, 3});
    auto inv_variance_tensor = Tensor::make_tensor<float>({1, 3});
    auto dscale_tensor = Tensor::make_tensor<float>({1, 3});
    auto dbias_tensor = Tensor::make_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<float, float, float> ref_impl;

    ref_impl.batchnorm_bwd(dy_tensor,
                           x_tensor,
                           mean_tensor,
                           inv_variance_tensor,
                           scale_tensor,
                           dx_tensor,
                           dscale_tensor,
                           dbias_tensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdSanityValidation)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 1, 2, 2};

    auto x_tensor = Tensor::make_tensor<double>(dims);
    auto dy_tensor = Tensor::make_tensor<double>(dims);
    auto dx_tensor = Tensor::make_tensor<double>(dims);
    auto scale_tensor = Tensor::make_tensor<double>({1, 1, 1, 1});
    auto mean_tensor = Tensor::make_tensor<double>({1, 1, 1, 1});
    auto inv_variance_tensor = Tensor::make_tensor<double>({1, 1, 1, 1});
    auto dscale_tensor = Tensor::make_tensor<double>({1, 1, 1, 1});
    auto dbias_tensor = Tensor::make_tensor<double>({1, 1, 1, 1});

    // x = [1, 2, 3, 4]
    x_tensor.set_host_value<double>(0, 0, 0, 0, 1.0);
    x_tensor.set_host_value<double>(0, 0, 0, 1, 2.0);
    x_tensor.set_host_value<double>(0, 0, 1, 0, 3.0);
    x_tensor.set_host_value<double>(0, 0, 1, 1, 4.0);

    // gradient dy = [0.1, 0.2, 0.3, 0.4]
    dy_tensor.set_host_value<double>(0, 0, 0, 0, 0.1);
    dy_tensor.set_host_value<double>(0, 0, 0, 1, 0.2);
    dy_tensor.set_host_value<double>(0, 0, 1, 0, 0.3);
    dy_tensor.set_host_value<double>(0, 0, 1, 1, 0.4);

    // scale (one channel) = 2.0
    scale_tensor.set_host_value<double>(0, 0, 0, 0, 2.0);

    // 1 batch, so compute mean and variance over all elements
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 5.0 / 4 = 1.25
    // inv_variance = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    mean_tensor.set_host_value<double>(0, 0, 0, 0, 2.5);
    inv_variance_tensor.set_host_value<double>(0, 0, 0, 0, 0.894423613312618);

    // dbias = sum(dy) = 0.1 + 0.2 + 0.3 + 0.4 = 1.0
    auto expected_dbias = 1.0;

    // x_hat = (x - mean) * inv_variance = [-1.34163542 -0.44721181  0.44721181  1.34163542]
    // dscale = sum(dy * x_hat) = sum([-1.34163542 -0.44721181  0.44721181  1.34163542]) = 0.447211806656309
    auto expected_dscale = 0.447211806656309;

    // dx is calculated pointwise via the full backward formula
    // dx = scale * inv_variance * (dy - mean(dy) - x_hat * dscale / 4)
    std::vector<double> expected_dx
        = {-2.14659950e-06, -7.15533166e-07, 7.15533166e-07, 2.14659950e-06};

    Cpu_fp_reference_implementation<double, double, double> ref_impl;
    ref_impl.batchnorm_bwd(dy_tensor,
                           x_tensor,
                           mean_tensor,
                           inv_variance_tensor,
                           scale_tensor,
                           dx_tensor,
                           dscale_tensor,
                           dbias_tensor);

    auto tolerance = 1e-6;

    EXPECT_NEAR(dbias_tensor.get_host_value<double>(0, 0, 0, 0), expected_dbias, tolerance);
    EXPECT_NEAR(dscale_tensor.get_host_value<double>(0, 0, 0, 0), expected_dscale, tolerance);
    EXPECT_NEAR(dx_tensor.get_host_value<double>(0, 0, 0, 0), expected_dx[0], tolerance);
    EXPECT_NEAR(dx_tensor.get_host_value<double>(0, 0, 0, 1), expected_dx[1], tolerance);
    EXPECT_NEAR(dx_tensor.get_host_value<double>(0, 0, 1, 0), expected_dx[2], tolerance);
    EXPECT_NEAR(dx_tensor.get_host_value<double>(0, 0, 1, 1), expected_dx[3], tolerance);
}
