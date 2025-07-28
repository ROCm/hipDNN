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

TEST(CpuFpReferenceImplementation, BasicFloatUsage)
{
    Tensor input_tensor = Tensor::make_nchw_tensor<float>({1, 3, 224, 224});
    Tensor output_tensor = Tensor::make_nchw_tensor<float>({1, 3, 224, 224});
    Tensor bias_tensor = Tensor::make_nchw_tensor<float>({1, 3});
    Tensor scale_tensor = Tensor::make_nchw_tensor<float>({1, 3});
    Tensor mean_tensor = Tensor::make_nchw_tensor<float>({1, 3});
    Tensor variance_tensor = Tensor::make_nchw_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<float, float, float> ref_impl;

    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BasicBFloat16Usage)
{
    Tensor input_tensor = Tensor::make_nchw_tensor<hip_bfloat16>({1, 3, 224, 224});
    Tensor output_tensor = Tensor::make_nchw_tensor<hip_bfloat16>({1, 3, 224, 224});
    Tensor bias_tensor = Tensor::make_nchw_tensor<float>({1, 3});
    Tensor scale_tensor = Tensor::make_nchw_tensor<float>({1, 3});
    Tensor mean_tensor = Tensor::make_nchw_tensor<float>({1, 3});
    Tensor variance_tensor = Tensor::make_nchw_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<hip_bfloat16, float, float> ref_impl;

    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BasicHalfUsage)
{
    Tensor input_tensor = Tensor::make_nchw_tensor<half>({1, 3, 224, 224});
    Tensor output_tensor = Tensor::make_nchw_tensor<half>({1, 3, 224, 224});
    Tensor bias_tensor = Tensor::make_nchw_tensor<float>({1, 3});
    Tensor scale_tensor = Tensor::make_nchw_tensor<float>({1, 3});
    Tensor mean_tensor = Tensor::make_nchw_tensor<float>({1, 3});
    Tensor variance_tensor = Tensor::make_nchw_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<half, float, float> ref_impl;

    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);
}

TEST(CpuFpReferenceImplementaion, BasicDoubleUsage)
{
    Tensor input_tensor = Tensor::make_nchw_tensor<double>({1, 3, 224, 224});
    Tensor output_tensor = Tensor::make_nchw_tensor<double>({1, 3, 224, 224});
    Tensor bias_tensor = Tensor::make_nchw_tensor<double>({1, 3});
    Tensor scale_tensor = Tensor::make_nchw_tensor<double>({1, 3});
    Tensor mean_tensor = Tensor::make_nchw_tensor<double>({1, 3});
    Tensor variance_tensor = Tensor::make_nchw_tensor<double>({1, 3});

    Cpu_fp_reference_implementation<double, double, double> ref_impl;

    ref_impl.batchnorm_fwd_inference(
        input_tensor, scale_tensor, bias_tensor, mean_tensor, variance_tensor, output_tensor, 1e-5);
}
