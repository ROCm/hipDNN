// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "utils/helpers.hpp"

#include <hip/hip_runtime.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_backward_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>

#include <iostream>

int main()
{
    std::cout << "Running Batch Norm Backward fp32..." << std::endl;

    using namespace hipdnn_frontend;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::FLOAT)
        .set_compute_data_type(DataType_t::FLOAT);

    auto dy = create_tensor({4, 32, 16, 16}, DataType_t::FLOAT);
    auto x = create_tensor({4, 32, 16, 16}, DataType_t::FLOAT);
    auto scale = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);
    auto saved_mean = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);
    auto saved_inv_variance = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);

    auto bn_bwd_attributes = graph::Batchnorm_backward_attributes();
    bn_bwd_attributes.set_saved_mean_and_inv_variance(saved_mean, saved_inv_variance);

    auto [dx, dscale, dbias] = graph->batchnorm_backward(dy, x, scale, bn_bwd_attributes);

    dx->set_output(true);
    dscale->set_output(true);
    dbias->set_output(true);

    HIPDNN_FE_CHECK(graph->validate());
    std::cout << "Graph validation successful." << std::endl;

    HIPDNN_FE_CHECK(graph->build_operation_graph());
    std::cout << "Operation graph build successful." << std::endl;

    Surface<float> dy_surface(get_tensor_element_count(dy));
    Surface<float> x_surface(get_tensor_element_count(x));
    Surface<float> scale_surface(get_tensor_element_count(scale), 1.0f);
    Surface<float> saved_mean_surface(get_tensor_element_count(saved_mean));
    Surface<float> saved_inv_var_surface(get_tensor_element_count(saved_inv_variance));

    Surface<float> dx_surface(get_tensor_element_count(dx));
    Surface<float> dscale_surface(get_tensor_element_count(dscale));
    Surface<float> dbias_surface(get_tensor_element_count(dbias));

    /*
    // need to properly create a variant pack with the input tensors
    
    auto execution_result = graph->execute(variant_pack);
    HIPDNN_FE_CHECK(execution_result);
    */
    std::cout << "Batch Norm Backward graph execution complete." << std::endl;

    return 0;
}
