// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../utils/helpers.hpp"

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_backward_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
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

    std::cout << "Running batch normalization backwards graph " << input_type << "...\n";

    int64_t N = 16; // BATCH SIZE
    int64_t C = 16; // CHANNELS (FEATURES)
    int64_t H = 16; // HEIGHT (SPATIAL DIMENSION)
    int64_t W = 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(input_type)
        .set_intermediate_data_type(intermediate_type)
        .set_compute_data_type(intermediate_type);

    auto dy = create_tensor({N, C, H, W}, input_type);
    auto x = create_tensor({N, C, H, W}, input_type);
    auto scale = create_tensor({1, C, 1, 1}, intermediate_type);
    auto saved_mean = create_tensor({1, C, 1, 1}, intermediate_type);
    auto saved_inv_variance = create_tensor({1, C, 1, 1}, intermediate_type);

    auto bn_bwd_attributes = graph::Batchnorm_backward_attributes();
    bn_bwd_attributes.set_saved_mean_and_inv_variance(saved_mean, saved_inv_variance);

    auto [dx, dscale, dbias] = graph->batchnorm_backward(dy, x, scale, bn_bwd_attributes);

    dx->set_output(true);
    dscale->set_output(true);
    dbias->set_output(true);

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

    auto dy_tensor = Tensor::make_nchw_tensor<InputType>(dy->get_dim());
    auto x_tensor = Tensor::make_nchw_tensor<InputType>(x->get_dim());
    auto scale_tensor = Tensor::make_nchw_tensor<IntermediateType>(scale->get_dim());
    auto saved_mean_tensor = Tensor::make_nchw_tensor<IntermediateType>(saved_mean->get_dim());
    auto saved_inv_var_tensor
        = Tensor::make_nchw_tensor<IntermediateType>(saved_inv_variance->get_dim());

    auto dx_tensor = Tensor::make_nchw_tensor<InputType>(dx->get_dim());
    auto dscale_tensor = Tensor::make_nchw_tensor<IntermediateType>(dscale->get_dim());
    auto dbias_tensor = Tensor::make_nchw_tensor<IntermediateType>(dbias->get_dim());

    dy_tensor.template fill_with_random_values<InputType>(static_cast<InputType>(0.0f),
                                                          static_cast<InputType>(1.0f));
    x_tensor.template fill_with_random_values<InputType>(static_cast<InputType>(0.0f),
                                                         static_cast<InputType>(1.0f));
    scale_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    saved_mean_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f));
    saved_inv_var_tensor.template fill_with_random_values<IntermediateType>(
        static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f));

    std::unordered_map<int64_t, void*> variant_pack;

    // TODO: Cleanup syntax when there is a better way to grab these pointers.
    variant_pack[dy->get_uid()] = dy_tensor.memory().template device_data<void>();
    variant_pack[x->get_uid()] = x_tensor.memory().template device_data<void>();
    variant_pack[scale->get_uid()] = scale_tensor.memory().template device_data<void>();
    variant_pack[saved_mean->get_uid()] = saved_mean_tensor.memory().template device_data<void>();
    variant_pack[saved_inv_variance->get_uid()]
        = saved_inv_var_tensor.memory().template device_data<void>();
    variant_pack[dx->get_uid()] = dx_tensor.memory().template device_data<void>();
    variant_pack[dscale->get_uid()] = dscale_tensor.memory().template device_data<void>();
    variant_pack[dbias->get_uid()] = dbias_tensor.memory().template device_data<void>();

    HIPDNN_FE_CHECK(graph->execute(handle, variant_pack, nullptr));

    dx_tensor.memory().mark_device_modified();
    auto dx_host_ptr = dx_tensor.memory().template host_data<InputType>();

    // TODO: Add CPU reference when available.
    // Example in bn_inference.cpp.

    std::cout << "First 10 dx values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(dx_host_ptr[i]) << " ";
    }

    std::cout << "\nBatch normalization backward graph execution complete for " << input_type
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
    std::cout << "All batch normalization backwards runs completed successfully.\n";
    return 0;
}
