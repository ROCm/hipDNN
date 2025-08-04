// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "utils/helpers.hpp"

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
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
void run_bn_inference(hipdnnHandle_t handle, const std::string& type_string)
{
    std::cout << "Running bnorm infer " << type_string << std::endl;

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

    auto mean = create_tensor({1, 32, 1, 1}, get_data_type<IntermediateType>());
    mean->set_uid(uid++);

    auto inv_variance = create_tensor({1, 32, 1, 1}, get_data_type<IntermediateType>());
    inv_variance->set_uid(uid++);

    auto bn_attributes = graph::Batchnorm_inference_attributes();
    bn_attributes.name = "bn_inference_node";

    auto y = graph->batchnorm_inference(x, mean, inv_variance, scale, bias, bn_attributes);
    y->set_output(true).set_data_type(get_data_type<InputType>());

    if(!y->has_uid())
    {
        y->set_uid(uid++);
    }

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

    auto x_tensor = Tensor::make_nchw_tensor<InputType>({4, 32, 16, 16});
    auto scale_tensor = Tensor::make_nchw_tensor<IntermediateType>({1, 32, 1, 1});
    auto bias_tensor = Tensor::make_nchw_tensor<IntermediateType>({1, 32, 1, 1});
    auto mean_tensor = Tensor::make_nchw_tensor<IntermediateType>({1, 32, 1, 1});
    auto inv_variance_tensor = Tensor::make_nchw_tensor<IntermediateType>({1, 32, 1, 1});
    auto y_tensor = Tensor::make_nchw_tensor<InputType>({4, 32, 16, 16});

    x_tensor.template fill_with_random_values<InputType>(static_cast<InputType>(0.0f),
                                                         static_cast<InputType>(1.0f));
    x_tensor.memory().mark_host_modified();

    scale_tensor.template fill_with_value<IntermediateType>(static_cast<IntermediateType>(1.0f));
    scale_tensor.memory().mark_host_modified();

    bias_tensor.template fill_with_value<IntermediateType>(static_cast<IntermediateType>(0.0f));
    bias_tensor.memory().mark_host_modified();

    mean_tensor.template fill_with_value<IntermediateType>(static_cast<IntermediateType>(0.5f));
    mean_tensor.memory().mark_host_modified();

    inv_variance_tensor.template fill_with_value<IntermediateType>(
        static_cast<IntermediateType>(1.0f));
    inv_variance_tensor.memory().mark_host_modified();

    std::unordered_map<int64_t, void*> variant_pack;
    variant_pack[x->get_uid()] = x_tensor.memory().template device_data<void>();
    variant_pack[scale->get_uid()] = scale_tensor.memory().template device_data<void>();
    variant_pack[bias->get_uid()] = bias_tensor.memory().template device_data<void>();
    variant_pack[mean->get_uid()] = mean_tensor.memory().template device_data<void>();
    variant_pack[inv_variance->get_uid()]
        = inv_variance_tensor.memory().template device_data<void>();
    variant_pack[y->get_uid()] = y_tensor.memory().template device_data<void>();

    HIPDNN_FE_CHECK(graph->execute(handle, variant_pack, nullptr));
    std::cout << "Graph execution successful." << std::endl;

    y_tensor.memory().mark_device_modified();
    auto y_host_ptr = y_tensor.memory().template host_data<InputType>();

    std::cout << "First 10 output values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(y_host_ptr[i]) << " ";
    }
    std::cout << std::endl;

    std::cout << "Bnorm infer graph execution complete for " << type_string << "." << std::endl
              << std::endl;
}

int main()
{
    hipdnn_frontend::initialize_frontend_logging(hipdnnLoggingCallback_ext);

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    run_bn_inference<float, float>(handle, "fp32");
    run_bn_inference<half, float>(handle, "fp16");
    run_bn_inference<hip_bfloat16, float>(handle, "bf16");

    HIPDNN_CHECK(hipdnnDestroy(handle));
    std::cout << "All tests completed successfully." << std::endl;
    return 0;
}