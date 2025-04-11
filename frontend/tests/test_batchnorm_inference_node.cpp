#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/node/batchnorm_inference_node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(BatchnormInferenceNodeTests, BatchnormInferenceNodeProperties)
{
    Batchnorm_inference_attributes batchnorm_attributes;
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());

    auto input_tensor = batchnorm_attributes.get_x();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = batchnorm_attributes.get_y();
    output_tensor->set_uid(2).set_name("OutputTensor");

    Graph_attributes       graph_attributes;
    BatchnormInferenceNode node(std::move(batchnorm_attributes), graph_attributes);
    auto                   error = node.infer_properties_node();

    EXPECT_EQ(error.code, error_code_t::OK);
    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}