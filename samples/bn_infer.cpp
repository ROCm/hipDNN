// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "utils/helpers.hpp"

#include <hip/hip_runtime.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>

#include <iostream>
#include <numeric>
#include <random>

int main()
{
    std::cout << "Running bnorm infer fp32" << std::endl;

    using namespace hipdnn_frontend;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::FLOAT)
        .set_compute_data_type(DataType_t::FLOAT);

    auto x = create_tensor({4, 32, 16, 16}, DataType_t::FLOAT);
    auto scale = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);
    auto bias = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);
    auto mean = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);
    auto inv_variance = create_tensor({1, 32, 1, 1}, DataType_t::FLOAT);

    auto bn_attributes = graph::Batchnorm_inference_attributes();
    bn_attributes.name = "bn_inference_node";

    auto y = graph->batchnorm_inference(x, mean, inv_variance, scale, bias, bn_attributes);
    y->set_output(true).set_data_type(DataType_t::FLOAT);

    auto validation_result = graph->validate();
    if(!validation_result.is_good())
    {
        std::cerr << "Graph validation failed: " << validation_result.get_message() << std::endl;
        return -1;
    }
    std::cout << "Graph validation successful." << std::endl;

    auto build_result = graph->build_operation_graph();
    if(!build_result.is_good())
    {
        std::cerr << "Building operation graph failed: " << build_result.get_message() << std::endl;
        return -1;
    }
    std::cout << "Operation graph build successful." << std::endl;

    Surface<float> x_surface(get_tensor_element_count(x));
    Surface<float> scale_surface(get_tensor_element_count(scale), 1.0f);
    Surface<float> bias_surface(get_tensor_element_count(bias), 0.0f);
    Surface<float> mean_surface(get_tensor_element_count(mean), 0.5f);
    Surface<float> inv_variance_surface(get_tensor_element_count(inv_variance), 1.0f);
    Surface<float> y_surface(get_tensor_element_count(y));

    // auto variant_pack_result = graph->create_variant_pack();
    // if(!variant_pack_result.is_good())
    // {
    //     std::cerr << "Failed to create variant pack: " << variant_pack_result.get_message()
    //               << std::endl;
    //     return -1;
    // }
    // auto variant_pack = variant_pack_result.get_data();
    // variant_pack->set_data_pointer(x->get_uid(), d_x)
    //     .set_data_pointer(scale->get_uid(), d_scale)
    //     .set_data_pointer(bias->get_uid(), d_bias)
    //     .set_data_pointer(mean->get_uid(), d_mean)
    //     .set_data_pointer(inv_variance->get_uid(), d_inv_variance)
    //     .set_data_pointer(y->get_uid(), d_y);

    // auto execution_result = graph->execute(variant_pack);
    // if(!execution_result.is_good())
    // {
    //     std::cerr << "Graph execution failed: " << execution_result.get_message() << std::endl;
    //     return -1;
    // }

    std::cout << "Bnorm infer graph execution complete." << std::endl;

    return 0;
}