// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "utils/helpers.hpp"

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

#include <iostream>
#include <random>
#include <string>
#include <unordered_map>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;

template <typename T>
inline DataType_t get_data_type();

template <>
inline DataType_t get_data_type<float>()
{
    return DataType_t::FLOAT;
}

template <>
inline DataType_t get_data_type<half>()
{
    return DataType_t::HALF;
}

template <>
inline DataType_t get_data_type<hip_bfloat16>()
{
    return DataType_t::BFLOAT16;
}

template <typename InputType, typename IntermediateType>
void run_bn_training(hipdnnHandle_t handle, const std::string& type_string)
{
    std::cout << "Running Batch Norm Training " << type_string << "..." << std::endl;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(get_data_type<InputType>())
        .set_intermediate_data_type(get_data_type<IntermediateType>())
        .set_compute_data_type(get_data_type<IntermediateType>());

    int64_t uid = 1;
    auto x = create_tensor({4, 32, 16, 16}, get_data_type<InputType>());
    x->set_uid(uid++);
    auto scale = create_tensor({1, 32, 1, 1}, get_data_type<IntermediateType>());
    scale->set_uid(uid++);
    auto bias = create_tensor({1, 32, 1, 1}, get_data_type<IntermediateType>());
    bias->set_uid(uid++);
    auto prev_running_mean = create_tensor({1, 32, 1, 1}, get_data_type<IntermediateType>());
    prev_running_mean->set_uid(uid++);
    auto prev_running_var = create_tensor({1, 32, 1, 1}, get_data_type<IntermediateType>());
    prev_running_var->set_uid(uid++);
    auto momentum = create_tensor({1, 1, 1, 1}, get_data_type<IntermediateType>());
    momentum->set_uid(uid++);
    auto epsilon = create_tensor({1, 1, 1, 1}, get_data_type<IntermediateType>());
    epsilon->set_uid(uid++);

    auto bn_attributes = graph::Batchnorm_attributes();
    bn_attributes.set_previous_running_stats(prev_running_mean, prev_running_var, momentum)
        .set_epsilon(epsilon);

    auto [y, next_running_mean, next_running_var, saved_mean, saved_inv_variance]
        = graph->batchnorm(x, scale, bias, bn_attributes);

    y->set_output(true).set_uid(uid++);
    next_running_mean->set_output(true).set_uid(uid++);
    next_running_var->set_output(true).set_uid(uid++);
    saved_mean->set_output(true).set_uid(uid++);
    saved_inv_variance->set_output(true).set_uid(uid++);

    HIPDNN_FE_CHECK(graph->validate());
    std::cout << "Graph validation successful." << std::endl;

    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));
    std::cout << "Operation graph build successful." << std::endl;

    HIPDNN_FE_CHECK(graph->create_execution_plans(handle));
    std::cout << "Execution plans created successfully." << std::endl;

    HIPDNN_FE_CHECK(graph->check_support());
    std::cout << "Graph support check successful." << std::endl;

    HIPDNN_FE_CHECK(graph->build_plans());
    std::cout << "Plans build successful." << std::endl;

    auto x_tensor = Tensor::make_nchw_tensor<InputType>(x->get_dim());
    auto scale_tensor = Tensor::make_nchw_tensor<IntermediateType>(scale->get_dim());
    auto bias_tensor = Tensor::make_nchw_tensor<IntermediateType>(bias->get_dim());
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
    x_tensor.memory().mark_host_modified();
    scale_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    scale_tensor.memory().mark_host_modified();
    bias_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    bias_tensor.memory().mark_host_modified();
    prev_mean_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    prev_mean_tensor.memory().mark_host_modified();
    prev_var_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f));
    prev_var_tensor.memory().mark_host_modified();

    momentum_tensor.memory().template host_data<IntermediateType>()[0] = 0.1f;
    momentum_tensor.memory().mark_host_modified();
    epsilon_tensor.memory().template host_data<IntermediateType>()[0] = 1e-5f;
    epsilon_tensor.memory().mark_host_modified();

    std::unordered_map<int64_t, void*> variant_pack;
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
    std::cout << "Graph execution successful." << std::endl;

    y_tensor.memory().mark_device_modified();
    auto y_host_ptr = y_tensor.memory().template host_data<InputType>();
    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(y_host_ptr[i]) << " ";
    }
    std::cout << std::endl;

    std::cout << "Batch Norm Training graph execution complete for " << type_string << "."
              << std::endl
              << std::endl;
}

int main()
{
    hipdnn_frontend::initialize_frontend_logging(hipdnnLoggingCallback_ext);

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    run_bn_training<float, float>(handle, "fp32");
    run_bn_training<half, float>(handle, "fp16");
    run_bn_training<hip_bfloat16, float>(handle, "bf16");

    HIPDNN_CHECK(hipdnnDestroy(handle));
    std::cout << "All tests completed successfully." << std::endl;
    return 0;
}