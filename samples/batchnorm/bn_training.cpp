// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../utils/helpers.hpp"

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

#include <iostream>
#include <string>
#include <unordered_map>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;

template <typename InputType, typename IntermediateType>
void run_bn_training(hipdnnHandle_t handle)
{
    auto input_type = get_data_type_enum_from_type<InputType>();
    auto intermediate_type = get_data_type_enum_from_type<IntermediateType>();

    std::cout << "Running batch normalization training graph " << input_type << "...\n";

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(input_type)
        .set_intermediate_data_type(intermediate_type)
        .set_compute_data_type(intermediate_type);

    int64_t uid = 1;
    auto x = create_tensor({16, 16, 16, 16}, input_type);
    x->set_uid(uid++);

    auto gamma = create_tensor({1, 16, 1, 1}, intermediate_type);
    gamma->set_uid(uid++);

    auto beta = create_tensor({1, 16, 1, 1}, intermediate_type);
    beta->set_uid(uid++);

    auto prev_running_mean = create_tensor({1, 16, 1, 1}, intermediate_type);
    prev_running_mean->set_uid(uid++);

    auto prev_running_var = create_tensor({1, 16, 1, 1}, intermediate_type);
    prev_running_var->set_uid(uid++);

    auto momentum = create_tensor({1, 1, 1, 1}, intermediate_type);
    momentum->set_uid(uid++);

    auto epsilon = create_tensor({1, 1, 1, 1}, intermediate_type);
    epsilon->set_uid(uid++);

    auto bn_attributes = graph::Batchnorm_attributes();
    bn_attributes.set_previous_running_stats(prev_running_mean, prev_running_var, momentum)
        .set_epsilon(epsilon);

    auto [y, next_running_mean, next_running_var, saved_mean, saved_inv_variance]
        = graph->batchnorm(x, gamma, beta, bn_attributes);

    y->set_output(true).set_uid(uid++);
    next_running_mean->set_output(true).set_uid(uid++);
    next_running_var->set_output(true).set_uid(uid++);
    saved_mean->set_output(true).set_uid(uid++);
    saved_inv_variance->set_output(true).set_uid(uid++);

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
    auto gamma_tensor = Tensor::make_nchw_tensor<IntermediateType>(gamma->get_dim());
    auto beta_tensor = Tensor::make_nchw_tensor<IntermediateType>(beta->get_dim());
    auto prev_mean_tensor
        = Tensor::make_nchw_tensor<IntermediateType>(prev_running_mean->get_dim());
    auto prev_var_tensor = Tensor::make_nchw_tensor<IntermediateType>(prev_running_var->get_dim());
    auto momentum_tensor = Tensor::make_nchw_tensor<IntermediateType>(momentum->get_dim());
    auto epsilon_tensor = Tensor::make_nchw_tensor<IntermediateType>(epsilon->get_dim());

    auto y_tensor = Tensor::make_nchw_tensor<InputType>(y->get_dim());
    auto next_mean_tensor
        = Tensor::make_nchw_tensor<IntermediateType>(next_running_mean->get_dim());
    auto next_var_tensor = Tensor::make_nchw_tensor<IntermediateType>(next_running_var->get_dim());
    auto saved_mean_tensor = Tensor::make_nchw_tensor<IntermediateType>(saved_mean->get_dim());
    auto saved_inv_var_tensor
        = Tensor::make_nchw_tensor<IntermediateType>(saved_inv_variance->get_dim());

    x_tensor.template fill_with_random_values<InputType>(static_cast<InputType>(0.0f),
                                                         static_cast<InputType>(1.0f));
    gamma_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    beta_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    prev_mean_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    prev_var_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f));

    momentum_tensor.memory().template host_data<IntermediateType>()[0] = 0.1f;
    epsilon_tensor.memory().template host_data<IntermediateType>()[0] = 1e-5f;

    std::unordered_map<int64_t, void*> variant_pack;
    variant_pack[x->get_uid()] = x_tensor.memory().template device_data<void>();
    variant_pack[gamma->get_uid()] = gamma_tensor.memory().template device_data<void>();
    variant_pack[beta->get_uid()] = beta_tensor.memory().template device_data<void>();
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
    auto y_host_ptr = y_tensor.memory().template host_data<InputType>();
    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(y_host_ptr[i]) << " ";
    }

    std::cout << "\nBatch normalization training graph execution complete for " << input_type
              << ".\n\n";
}

int main()
{
    initialize_frontend_logging(hipdnnLoggingCallback_ext);

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    run_bn_training<float, float>(handle);
    run_bn_training<half, float>(handle);
    run_bn_training<hip_bfloat16, float>(handle);

    HIPDNN_CHECK(hipdnnDestroy(handle));
    std::cout << "All batch normalization training runs completed successfully.\n";
    return 0;
}