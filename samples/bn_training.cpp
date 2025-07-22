// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "utils/helpers.hpp"

#include <hip/hip_runtime.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>

#include <iostream>

int main()
{
    std::cout << "Running Batch Norm Training fp32..." << std::endl;

    using namespace hipdnn_frontend;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::FLOAT)
        .set_compute_data_type(DataType_t::FLOAT);

    auto x = create_tensor({4, 32, 16, 16}, DataType_t::FLOAT);
    auto scale = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);
    auto bias = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);
    auto prev_running_mean = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);
    auto prev_running_var = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);
    auto momentum = create_tensor({1, 1, 1, 1}, DataType_t::FLOAT);
    auto epsilon = create_tensor({1, 1, 1, 1}, DataType_t::FLOAT);

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
    std::cout << "Graph validation successful." << std::endl;

    HIPDNN_FE_CHECK(graph->build_operation_graph());
    std::cout << "Operation graph build successful." << std::endl;

    Surface<float> x_surface(get_tensor_element_count(x));
    Surface<float> scale_surface(get_tensor_element_count(scale), 1.0f);
    Surface<float> bias_surface(get_tensor_element_count(bias), 0.0f);
    Surface<float> prev_mean_surface(get_tensor_element_count(prev_running_mean), 0.0f);
    Surface<float> prev_var_surface(get_tensor_element_count(prev_running_var), 1.0f);
    Surface<float> momentum_surface(get_tensor_element_count(momentum), 0.1f);

    Surface<float> y_surface(get_tensor_element_count(y));
    Surface<float> next_mean_surface(get_tensor_element_count(next_running_mean));
    Surface<float> next_var_surface(get_tensor_element_count(next_running_var));
    Surface<float> saved_mean_surface(get_tensor_element_count(saved_mean));
    Surface<float> saved_inv_var_surface(get_tensor_element_count(saved_inv_variance));

    /*
    // need to properly create a variant pack with the input tensors

    auto execution_result = graph->execute(variant_pack);
    HIPDNN_FE_CHECK(execution_result);
    */
    std::cout << "Batch Norm Training graph execution complete." << std::endl;

    return 0;
}
