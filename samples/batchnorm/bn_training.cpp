// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../utils/helpers.hpp"

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

#include <iostream>
#include <string>
#include <unordered_map>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;

// TODO: verify this sample when applicable engines are added
template <typename InputType, typename IntermediateType>
void Sample_runner::operator()(const Tensor_layout& layout)
{
    auto input_type = get_data_type_enum_from_type<InputType>();
    auto intermediate_type = get_data_type_enum_from_type<IntermediateType>();

    std::cout << "Running batch normalization training graph " << input_type << " [" << layout
              << "]" << (config.cpu_validation ? " (with CPU validation)" : "") << "...\n";

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
    auto prev_running_mean = create_tensor({1, C, 1, 1}, intermediate_type);
    auto prev_running_var = create_tensor({1, C, 1, 1}, intermediate_type);
    auto momentum = create_tensor({1, 1, 1, 1}, intermediate_type);
    auto epsilon = create_tensor({1, 1, 1, 1}, intermediate_type);
    auto bn_attributes = graph::Batchnorm_attributes();
    bn_attributes.set_previous_running_stats(prev_running_mean, prev_running_var, momentum)
        .set_epsilon(epsilon);

    auto [y, next_running_mean, next_running_var, saved_mean, saved_inv_variance]
        = graph->batchnorm(x, scale, bias, bn_attributes);

    y->set_output(true);
    next_running_mean->set_output(true);
    next_running_var->set_output(true);
    saved_mean->set_output(true);
    saved_inv_variance->set_output(true);

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

    auto x_tensor = Tensor::make_tensor<InputType>(x->get_dim(), layout);
    auto scale_tensor = Tensor::make_tensor<IntermediateType>(scale->get_dim());
    auto bias_tensor = Tensor::make_tensor<IntermediateType>(bias->get_dim());
    auto prev_mean_tensor = Tensor::make_tensor<IntermediateType>(prev_running_mean->get_dim());
    auto prev_var_tensor = Tensor::make_tensor<IntermediateType>(prev_running_var->get_dim());
    auto momentum_tensor = Tensor::make_tensor<IntermediateType>(momentum->get_dim());
    auto epsilon_tensor = Tensor::make_tensor<IntermediateType>(epsilon->get_dim());

    auto y_tensor = Tensor::make_tensor<InputType>(y->get_dim(), layout);
    auto next_mean_tensor = Tensor::make_tensor<IntermediateType>(next_running_mean->get_dim());
    auto next_var_tensor = Tensor::make_tensor<IntermediateType>(next_running_var->get_dim());
    auto saved_mean_tensor = Tensor::make_tensor<IntermediateType>(saved_mean->get_dim());
    auto saved_inv_var_tensor
        = Tensor::make_tensor<IntermediateType>(saved_inv_variance->get_dim());

    x_tensor.template fill_with_random_values<InputType>(static_cast<InputType>(0.0f),
                                                         static_cast<InputType>(1.0f));
    scale_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    bias_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    prev_mean_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    prev_var_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f));

    momentum_tensor.memory().template host_data<IntermediateType>()[0] = 0.1f;
    epsilon_tensor.memory().template host_data<IntermediateType>()[0] = 1e-5f;

    std::unordered_map<int64_t, void*> variant_pack;

    // TODO: Cleanup syntax when there is a better way to grab these pointers.
    variant_pack[x->get_uid()] = x_tensor.memory().template device_data<void>();
    variant_pack[scale->get_uid()] = scale_tensor.memory().template device_data<void>();
    variant_pack[bias->get_uid()] = bias_tensor.memory().template device_data<void>();
    variant_pack[prev_running_mean->get_uid()]
        = prev_mean_tensor.memory().template device_data<void>();
    variant_pack[prev_running_var->get_uid()]
        = prev_var_tensor.memory().template device_data<void>();
    variant_pack[momentum->get_uid()] = momentum_tensor.memory().template device_data<void>();
    variant_pack[epsilon->get_uid()] = epsilon_tensor.memory().template device_data<void>();
    variant_pack[y->get_uid()] = y_tensor.memory().template device_data<void>();
    variant_pack[next_running_mean->get_uid()]
        = next_mean_tensor.memory().template device_data<void>();
    variant_pack[next_running_var->get_uid()]
        = next_var_tensor.memory().template device_data<void>();
    variant_pack[saved_mean->get_uid()] = saved_mean_tensor.memory().template device_data<void>();
    variant_pack[saved_inv_variance->get_uid()]
        = saved_inv_var_tensor.memory().template device_data<void>();

    HIPDNN_FE_CHECK(graph->execute(handle, variant_pack, nullptr));

    y_tensor.memory().mark_device_modified();
    next_mean_tensor.memory().mark_device_modified();
    next_var_tensor.memory().mark_device_modified();
    saved_mean_tensor.memory().mark_device_modified();
    saved_inv_var_tensor.memory().mark_device_modified();

    auto y_host_ptr = y_tensor.memory().template host_data<InputType>();

    if(config.cpu_validation)
    {
        std::cout << "Running CPU reference validation...\n";

        auto ref_impl = hipdnn_sdk::reference_test_utilities::
            Cpu_fp_reference_implementation<InputType, IntermediateType>();

        auto y_ref_tensor = Tensor::make_tensor<InputType>(y->get_dim(), layout);
        auto next_mean_ref_tensor
            = Tensor::make_tensor<IntermediateType>(next_running_mean->get_dim());
        auto next_var_ref_tensor
            = Tensor::make_tensor<IntermediateType>(next_running_var->get_dim());
        auto saved_mean_ref_tensor = Tensor::make_tensor<IntermediateType>(saved_mean->get_dim());
        auto saved_inv_var_ref_tensor
            = Tensor::make_tensor<IntermediateType>(saved_inv_variance->get_dim());

        // TODO: Uncomment when CPU reference implemented
        // ref_impl.batchnorm_fwd_training(x_tensor,
        //                                scale_tensor,
        //                                bias_tensor,
        //                                prev_mean_tensor,
        //                                prev_var_tensor,
        //                                momentum_tensor,
        //                                epsilon_tensor,
        //                                y_ref_tensor,
        //                                next_mean_ref_tensor,
        //                                next_var_ref_tensor,
        //                                saved_mean_ref_tensor,
        //                                saved_inv_var_ref_tensor);

        // auto epsilon = get_epsilon<InputType>();
        //
        // auto y_validator
        //     = hipdnn_sdk::reference_test_utilities::Cpu_fp_reference_validation<InputType>(
        //         static_cast<InputType>(epsilon), static_cast<InputType>(epsilon));
        //
        // auto stats_validator
        //     = hipdnn_sdk::reference_test_utilities::Cpu_fp_reference_validation<IntermediateType>(
        //         static_cast<IntermediateType>(epsilon), static_cast<IntermediateType>(epsilon));
        // bool y_valid = y_validator.compare_buffers(y_ref_tensor.memory(), y_tensor.memory());
        // bool next_mean_valid = stats_validator.compare_buffers(next_mean_ref_tensor.memory(),
        //                                                        next_mean_tensor.memory());
        // bool next_var_valid = stats_validator.compare_buffers(next_var_ref_tensor.memory(),
        //                                                       next_var_tensor.memory());
        // TODO: consider adding validation for other output buffers, but they are verified indirectly by y
        // std::cout << "CPU reference validation:\n";
        // std::cout << "  y: " << (y_valid ? "successful" : "failed") << "\n";
        // std::cout << "  next_running_mean: " << (next_mean_valid ? "successful" : "failed") << "\n";
        // std::cout << "  next_running_var: " << (next_var_valid ? "successful" : "failed") << "\n";

        std::cout << "CPU reference validation skipped - batchnorm training forward not yet "
                     "implemented.\n";
    }

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(y_host_ptr[i]) << " ";
    }

    std::cout << "\nBatch normalization training graph execution complete for " << input_type
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
    std::cout << "All batch normalization training runs completed successfully.\n";
    return 0;
}
