// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../utils/helpers.hpp"

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

#include <iostream>
#include <string>
#include <unordered_map>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;

template <typename InputType, typename IntermediateType, Tensor_layout Layout>
void Sample_runner::operator()()
{
    if constexpr(Layout == Tensor_layout::NHWC)
    {
        std::cout << "NHWC not supported yet\n";
        return;
    }

    auto input_type = get_data_type_enum_from_type<InputType>();
    auto intermediate_type = get_data_type_enum_from_type<IntermediateType>();

    std::cout << "Running batch normalization inference graph " << input_type
              << (config.cpu_validation ? " (with CPU validation)" : "") << "...\n";

    int64_t N = 16; // BATCH SIZE
    int64_t C = 16; // CHANNELS (FEATURES)
    int64_t H = 16; // HEIGHT (SPATIAL DIMENSION)
    int64_t W = 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(input_type)
        .set_intermediate_data_type(intermediate_type)
        .set_compute_data_type(intermediate_type);

    auto x = create_tensor({N, C, H, W}, input_type);
    auto scale = create_tensor({1, C, 1, 1}, intermediate_type);
    auto bias = create_tensor({1, C, 1, 1}, intermediate_type);
    auto mean = create_tensor({1, C, 1, 1}, intermediate_type);
    auto inv_variance = create_tensor({1, C, 1, 1}, intermediate_type);

    auto bn_attributes = graph::Batchnorm_inference_attributes();
    bn_attributes.name = "bn_inference_node";

    auto y = graph->batchnorm_inference(x, mean, inv_variance, scale, bias, bn_attributes);
    y->set_output(true).set_data_type(input_type);

    HIPDNN_FE_CHECK(graph->validate());
    std::cout << "Graph validation successful.\n";

    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));
    std::cout << "Operation graph build successful.\n";

    HIPDNN_FE_CHECK(graph->create_execution_plans(handle));
    std::cout << "Execution plans created successfully.\n";

    HIPDNN_FE_CHECK(graph->check_support());
    std::cout << "Graph support check successful.\n";

    HIPDNN_FE_CHECK(graph->build_plans());
    std::cout << "Plans build successful.\n";

    auto x_tensor = Tensor::make_nchw_tensor<InputType>(x->get_dim());
    auto scale_tensor = Tensor::make_nchw_tensor<IntermediateType>(scale->get_dim());
    auto bias_tensor = Tensor::make_nchw_tensor<IntermediateType>(bias->get_dim());
    auto mean_tensor = Tensor::make_nchw_tensor<IntermediateType>(mean->get_dim());
    auto inv_variance_tensor = Tensor::make_nchw_tensor<IntermediateType>(inv_variance->get_dim());
    auto y_tensor = Tensor::make_nchw_tensor<InputType>(y->get_dim());

    x_tensor.template fill_with_random_values<InputType>(static_cast<InputType>(0.0f),
                                                         static_cast<InputType>(1.0f));

    scale_tensor.template fill_with_value<IntermediateType>(static_cast<IntermediateType>(1.0f));

    bias_tensor.template fill_with_value<IntermediateType>(static_cast<IntermediateType>(0.0f));

    mean_tensor.template fill_with_value<IntermediateType>(static_cast<IntermediateType>(0.5f));

    inv_variance_tensor.template fill_with_value<IntermediateType>(
        static_cast<IntermediateType>(1.0f));

    std::unordered_map<int64_t, void*> variant_pack;

    // TODO: Cleanup syntax when there is a better way to grab these pointers.
    variant_pack[x->get_uid()] = x_tensor.memory().template device_data<void>();
    variant_pack[scale->get_uid()] = scale_tensor.memory().template device_data<void>();
    variant_pack[bias->get_uid()] = bias_tensor.memory().template device_data<void>();
    variant_pack[mean->get_uid()] = mean_tensor.memory().template device_data<void>();
    variant_pack[inv_variance->get_uid()]
        = inv_variance_tensor.memory().template device_data<void>();
    variant_pack[y->get_uid()] = y_tensor.memory().template device_data<void>();

    HIPDNN_FE_CHECK(graph->execute(handle, variant_pack, nullptr));

    y_tensor.memory().mark_device_modified();
    auto y_host_ptr = y_tensor.memory().template host_data<InputType>();

    if(config.cpu_validation)
    {
        std::cout << "Running CPU reference validation...\n";

        auto ref_impl = hipdnn_sdk::reference_test_utilities::
            Cpu_fp_reference_implementation<InputType, IntermediateType>();
        auto y_ref_tensor = Tensor::make_nchw_tensor<InputType>(y->get_dim());

        // Convert inverse variance to variance for CPU reference
        auto variance_tensor = Tensor::make_nchw_tensor<IntermediateType>(inv_variance->get_dim());
        auto inv_variance_host_ptr
            = inv_variance_tensor.memory().template host_data<IntermediateType>();
        auto variance_host_ptr = variance_tensor.memory().template host_data<IntermediateType>();

        for(size_t i = 0; i < inv_variance_tensor.memory().count(); ++i)
        {
            variance_host_ptr[i] = static_cast<IntermediateType>(1.0f)
                                   / (inv_variance_host_ptr[i] * inv_variance_host_ptr[i]);
        }

        auto epsilon = 1e-2f; // bf16 fails for lower epsilon values

        ref_impl.batchnorm_fwd_inference(x_tensor,
                                         scale_tensor,
                                         bias_tensor,
                                         mean_tensor,
                                         variance_tensor,
                                         y_ref_tensor,
                                         epsilon);

        auto validator
            = hipdnn_sdk::reference_test_utilities::Cpu_fp_reference_validation<InputType>(
                static_cast<InputType>(epsilon), static_cast<InputType>(epsilon));

        std::cout << "CPU reference validation "
                  << (validator.compare_buffers(y_ref_tensor.memory(), y_tensor.memory())
                          ? "successful"
                          : "failed")
                  << ".\n";
    }

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(y_host_ptr[i]) << " ";
    }

    std::cout << "\nBatch normalization inference graph execution complete for " << input_type
              << ".\n\n";
}

int main(int argc, char* argv[])
{
    auto config = parse_command_line_args(argc, argv);

    initialize_frontend_logging(hipdnnLoggingCallback_ext);

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    run(Sample_runner{handle, config});

    HIPDNN_CHECK(hipdnnDestroy(handle));
    std::cout << "All batch normalization inference runs completed successfully.\n";
    return 0;
}
