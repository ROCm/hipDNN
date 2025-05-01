// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <iostream>
#include <memory>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

template <typename T>
static bool not_equal(const T& expected, const T& actual)
{
    return expected != actual;
}

static bool not_equal(const Tensor_attributes& tensor,
                      const hipdnn_sdk::data_objects::TensorAttributesT& serialized_tensor)
{
    return not_equal(tensor.get_name(), serialized_tensor.name)
           || not_equal(tensor.get_uid(), serialized_tensor.uid)
           || not_equal(to_sdk_type(tensor.get_data_type()), serialized_tensor.data_type)
           || not_equal(tensor.get_dim(), serialized_tensor.dims)
           || not_equal(tensor.get_stride(), serialized_tensor.strides);
}

int main()
{
    Graph graph;

    graph.set_name("SerializedBatchnormGraph")
        .set_compute_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::HALF)
        .set_io_data_type(DataType_t::FLOAT);

    auto x = std::make_shared<Tensor_attributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    auto scale = std::make_shared<Tensor_attributes>();
    scale->set_uid(2).set_name("Scale").set_data_type(DataType_t::FLOAT);

    auto bias = std::make_shared<Tensor_attributes>();
    bias->set_uid(3).set_name("Bias").set_data_type(DataType_t::FLOAT);

    auto prev_running_mean = std::make_shared<Tensor_attributes>();
    prev_running_mean->set_uid(4).set_name("PrevRunningMean").set_data_type(DataType_t::FLOAT);

    auto prev_running_variance = std::make_shared<Tensor_attributes>();
    prev_running_variance->set_uid(5)
        .set_name("PrevRunningVariance")
        .set_data_type(DataType_t::FLOAT);

    auto momentum = std::make_shared<Tensor_attributes>();
    momentum->set_uid(6).set_name("Momentum").set_data_type(DataType_t::FLOAT);

    auto epsilon = std::make_shared<Tensor_attributes>();
    epsilon->set_uid(7).set_name("Epsilon").set_data_type(DataType_t::FLOAT);

    Batchnorm_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormNode";
    batchnorm_attributes.set_previous_running_stats(
        prev_running_mean, prev_running_variance, momentum);
    batchnorm_attributes.set_epsilon(epsilon);

    auto [y, mean, inv_variance, next_running_mean, next_running_variance]
        = graph.batchnorm(x, scale, bias, batchnorm_attributes);

    auto validation_result = graph.validate();
    if(validation_result.is_bad())
    {
        std::cout << "Graph validation failed: " << validation_result.get_message() << std::endl;
        return EXIT_FAILURE;
    }

    auto build_result = graph.build_operation_graph();
    if(build_result.is_bad())
    {
        std::cout << "Graph build failed: " << build_result.get_message() << std::endl;
        return EXIT_FAILURE;
    }

    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(graph.serialized_graph.data());

    if(deserialized_graph == nullptr)
    {
        std::cout << "Graph unpacking failed" << std::endl;
        return EXIT_FAILURE;
    }

    if(not_equal(deserialized_graph->name, std::string("SerializedBatchnormGraph")))
    {
        std::cout << "Graph name mismatch: expected SerializedBatchnormGraph, got "
                  << deserialized_graph->name << std::endl;
        return EXIT_FAILURE;
    }

    if(not_equal(deserialized_graph->compute_type, hipdnn_sdk::data_objects::DataType_FLOAT))
    {
        std::cout << "Graph compute type mismatch: expected DataType_FLOAT, got "
                  << (int)deserialized_graph->compute_type << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(deserialized_graph->intermediate_type, hipdnn_sdk::data_objects::DataType_HALF))
    {
        std::cout << "Graph intermediate type mismatch: expected HALF, got "
                  << deserialized_graph->intermediate_type << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(deserialized_graph->io_type, hipdnn_sdk::data_objects::DataType_FLOAT))
    {
        std::cout << "Graph IO type mismatch: expected FLOAT, got " << deserialized_graph->io_type
                  << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(deserialized_graph->tensors.size(), static_cast<size_t>(12)))
    {
        std::cout << "Graph tensor size mismatch: expected 6, got "
                  << deserialized_graph->tensors.size() << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(deserialized_graph->nodes.size(), static_cast<size_t>(1)))
    {
        std::cout << "Graph node size mismatch: expected 1, got "
                  << deserialized_graph->nodes.size() << std::endl;
        return EXIT_FAILURE;
    }

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensor_lookup;
    for(auto& tensor : deserialized_graph->tensors)
    {
        tensor_lookup[tensor->uid] = *tensor;
    }

    if(not_equal(*x, tensor_lookup[x->get_uid()]))
    {
        std::cout << "Validation failed for tensor: X" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*scale, tensor_lookup[scale->get_uid()]))
    {
        std::cout << "Validation failed for tensor: Scale" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*bias, tensor_lookup[bias->get_uid()]))
    {
        std::cout << "Validation failed for tensor: Bias" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*epsilon, tensor_lookup[epsilon->get_uid()]))
    {
        std::cout << "Validation failed for tensor: Epsilon" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*prev_running_mean, tensor_lookup[prev_running_mean->get_uid()]))
    {
        std::cout << "Validation failed for tensor: PrevRunningMean" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*prev_running_variance, tensor_lookup[prev_running_variance->get_uid()]))
    {
        std::cout << "Validation failed for tensor: PrevRunningVariance" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*momentum, tensor_lookup[momentum->get_uid()]))
    {
        std::cout << "Validation failed for tensor: Momentum" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*y, tensor_lookup[y->get_uid()]))
    {
        std::cout << "Validation failed for tensor: Y" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*mean, tensor_lookup[mean->get_uid()]))
    {
        std::cout << "Validation failed for tensor: Mean" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*inv_variance, tensor_lookup[inv_variance->get_uid()]))
    {
        std::cout << "Validation failed for tensor: InvVariance" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*next_running_mean, tensor_lookup[next_running_mean->get_uid()]))
    {
        std::cout << "Validation failed for tensor: NextRunningMean" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(*next_running_variance, tensor_lookup[next_running_variance->get_uid()]))
    {
        std::cout << "Validation failed for tensor: NextRunningVariance" << std::endl;
        return EXIT_FAILURE;
    }

    if(not_equal(deserialized_graph->nodes[0]->attributes.type,
                 hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormAttributes))
    {
        std::cout << "Graph node type mismatch: expected NodeAttributes_BatchnormAttributes, got "
                  << deserialized_graph->nodes[0]->attributes.type << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(deserialized_graph->nodes[0]->name, std::string("BatchnormNode")))
    {
        std::cout << "Graph node name mismatch: expected BatchnormNode, got "
                  << deserialized_graph->nodes[0]->name << std::endl;
        return EXIT_FAILURE;
    }

    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[0]->attributes.AsBatchnormAttributes();

    if(not_equal(deserialized_batchnorm_attributes->x, x->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: x" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(deserialized_batchnorm_attributes->scale, scale->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: scale" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(deserialized_batchnorm_attributes->bias, bias->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: bias" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(deserialized_batchnorm_attributes->epsilon, epsilon->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: epsilon" << std::endl;
        return EXIT_FAILURE;
    }
    if(!deserialized_batchnorm_attributes->prev_running_mean.has_value()
       || not_equal(deserialized_batchnorm_attributes->prev_running_mean.value(),
                    prev_running_mean->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: prev_running_mean" << std::endl;
        return EXIT_FAILURE;
    }
    if(!deserialized_batchnorm_attributes->prev_running_variance.has_value()
       || not_equal(deserialized_batchnorm_attributes->prev_running_variance.value(),
                    prev_running_variance->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: prev_running_variance" << std::endl;
        return EXIT_FAILURE;
    }
    if(!deserialized_batchnorm_attributes->momentum.has_value()
       || not_equal(deserialized_batchnorm_attributes->momentum.value(), momentum->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: momentum" << std::endl;
        return EXIT_FAILURE;
    }
    if(not_equal(deserialized_batchnorm_attributes->y, y->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: y" << std::endl;
        return EXIT_FAILURE;
    }
    if(!deserialized_batchnorm_attributes->mean.has_value()
       || not_equal(deserialized_batchnorm_attributes->mean.value(), mean->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: mean" << std::endl;
        return EXIT_FAILURE;
    }
    if(!deserialized_batchnorm_attributes->inv_variance.has_value()
       || not_equal(deserialized_batchnorm_attributes->inv_variance.value(),
                    inv_variance->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: inv_variance" << std::endl;
        return EXIT_FAILURE;
    }
    if(!deserialized_batchnorm_attributes->next_running_mean.has_value()
       || not_equal(deserialized_batchnorm_attributes->next_running_mean.value(),
                    next_running_mean->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: next_running_mean" << std::endl;
        return EXIT_FAILURE;
    }
    if(!deserialized_batchnorm_attributes->next_running_variance.has_value()
       || not_equal(deserialized_batchnorm_attributes->next_running_variance.value(),
                    next_running_variance->get_uid()))
    {
        std::cout << "Batchnorm attribute mismatch: next_running_variance" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Graph serialization and deserialization successful!" << std::endl;

    return EXIT_SUCCESS;
}