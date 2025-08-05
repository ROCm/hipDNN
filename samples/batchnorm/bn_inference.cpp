// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../utils/helpers.hpp"

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

#include <iostream>
#include <string>
#include <unordered_map>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;

template <typename InputType, typename IntermediateType>
void run_bn_inference(hipdnnHandle_t handle)
{
    auto input_type = get_data_type_enum_from_type<InputType>();
    auto intermediate_type = get_data_type_enum_from_type<IntermediateType>();

    std::cout << "Running batch normalization inference graph " << input_type << "...\n";

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

    auto mean = create_tensor({1, 16, 1, 1}, intermediate_type);
    mean->set_uid(uid++);

    auto inv_variance = create_tensor({1, 16, 1, 1}, intermediate_type);
    inv_variance->set_uid(uid++);

    auto bn_attributes = graph::Batchnorm_inference_attributes();
    bn_attributes.name = "bn_inference_node";

    auto y = graph->batchnorm_inference(x, mean, inv_variance, gamma, beta, bn_attributes);
    y->set_output(true).set_data_type(input_type);

    if(!y->has_uid())
    {
        y->set_uid(uid++);
    }

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
    auto mean_tensor = Tensor::make_nchw_tensor<IntermediateType>(mean->get_dim());
    auto inv_variance_tensor = Tensor::make_nchw_tensor<IntermediateType>(inv_variance->get_dim());
    auto y_tensor = Tensor::make_nchw_tensor<InputType>(y->get_dim());

    x_tensor.template fill_with_random_values<InputType>(static_cast<InputType>(0.0f),
                                                         static_cast<InputType>(1.0f));

    gamma_tensor.template fill_with_value<IntermediateType>(static_cast<IntermediateType>(1.0f));

    beta_tensor.template fill_with_value<IntermediateType>(static_cast<IntermediateType>(0.0f));

    mean_tensor.template fill_with_value<IntermediateType>(static_cast<IntermediateType>(0.5f));

    inv_variance_tensor.template fill_with_value<IntermediateType>(
        static_cast<IntermediateType>(1.0f));

    std::unordered_map<int64_t, void*> variant_pack;
    variant_pack[x->get_uid()] = x_tensor.memory().template device_data<void>();
    variant_pack[gamma->get_uid()] = gamma_tensor.memory().template device_data<void>();
    variant_pack[beta->get_uid()] = beta_tensor.memory().template device_data<void>();
    variant_pack[mean->get_uid()] = mean_tensor.memory().template device_data<void>();
    variant_pack[inv_variance->get_uid()]
        = inv_variance_tensor.memory().template device_data<void>();
    variant_pack[y->get_uid()] = y_tensor.memory().template device_data<void>();

    HIPDNN_FE_CHECK(graph->execute(handle, variant_pack, nullptr));

    y_tensor.memory().mark_device_modified();
    auto y_host_ptr = y_tensor.memory().template host_data<InputType>();

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(y_host_ptr[i]) << " ";
    }

    std::cout << "\nBatch normalization inference graph execution complete for " << input_type
              << ".\n\n";
}

int main()
{
    initialize_frontend_logging(hipdnnLoggingCallback_ext);

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    run_bn_inference<float, float>(handle);
    run_bn_inference<half, float>(handle);
    run_bn_inference<hip_bfloat16, float>(handle);

    HIPDNN_CHECK(hipdnnDestroy(handle));
    std::cout << "All batch normalization inference runs completed successfully.\n";
    return 0;
}