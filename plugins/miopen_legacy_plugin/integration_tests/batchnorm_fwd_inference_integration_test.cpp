// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

//#include <cmath>
#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_frontend/attributes/tensor_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>
#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_sdk::utilities;

// NOLINTBEGIN
class BatchnormForwardInferenceIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        setenv("HIPDNN_LOG_LEVEL", "info", 1);
        //setenv("HIPDNN_LOG_FILE", "off", 1);
        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&device_id), hipSuccess);
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

        // Create handle
        ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(handle)
        {
            ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
        }
        if(stream)
        {
            ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        }
    }

    struct TensorShape
    {
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
        std::string name;
    };

    //todo, deal with the data types in a better way.
    template <typename Input_type, typename Intermediate_type>
    void RunBatchnormTest(const TensorShape& input_shape,
                          hipdnn_frontend::DataType_t data_type
                          = hipdnn_frontend::DataType_t::FLOAT,
                          double tolerance = 1e-4)
    {
        std::ignore = tolerance; // Unused parameter, can be used for validation later

        // Calculate channel dimension (assumes NCHW format)
        ASSERT_GE(input_shape.dims.size(), 2);
        int64_t channels = input_shape.dims[1];
        std::vector<int64_t> channel_dims = {1, channels, 1, 1};
        std::vector<int64_t> channel_strides = {channels, 1, channels, channels};

        // Create graph
        auto graph = std::make_shared<Graph>();
        graph->set_name("BatchnormInferenceTest")
            .set_io_data_type(data_type)
            .set_intermediate_data_type(data_type)
            .set_compute_data_type(data_type);

        auto x_tensor_attr = std::make_shared<Tensor_attributes>();
        x_tensor_attr->set_uid(1)
            .set_name("X")
            .set_data_type(data_type)
            .set_dim(input_shape.dims)
            .set_stride(input_shape.strides);

        auto mean_tensor_attr = std::make_shared<Tensor_attributes>();
        mean_tensor_attr->set_uid(2)
            .set_name("mean")
            .set_data_type(data_type)
            .set_dim(channel_dims)
            .set_stride(channel_strides);

        auto inv_variance_tensor_attr = std::make_shared<Tensor_attributes>();
        inv_variance_tensor_attr->set_uid(3)
            .set_name("inv_variance")
            .set_data_type(data_type)
            .set_dim(channel_dims)
            .set_stride(channel_strides);

        auto scale_tensor_attr = std::make_shared<Tensor_attributes>();
        scale_tensor_attr->set_uid(4)
            .set_name("scale")
            .set_data_type(data_type)
            .set_dim(channel_dims)
            .set_stride(channel_strides);

        auto bias_tensor_attr = std::make_shared<Tensor_attributes>();
        bias_tensor_attr->set_uid(5)
            .set_name("bias")
            .set_data_type(data_type)
            .set_dim(channel_dims)
            .set_stride(channel_strides);

        // Create batchnorm inference operation
        Batchnorm_inference_attributes bn_attrs;
        bn_attrs.set_name("batchnorm_inference");

        auto y_tensor_attr = graph->batchnorm_inference(x_tensor_attr,
                                                        mean_tensor_attr,
                                                        inv_variance_tensor_attr,
                                                        scale_tensor_attr,
                                                        bias_tensor_attr,
                                                        bn_attrs);

        // Validate and build graph
        auto result = graph->validate();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->build_operation_graph(handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->create_execution_plans(handle); //no engines found now.
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        //I cant build plans yet because I cant finalize the engine config descriptor due to
        // not getting the workspace size from the plugin resource manager.  Getting backend error
        // Failed to finalize engine config descriptor Backend error: _Map_base::at
        result = graph->build_plans();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        Tensor x_tensor = Tensor::make_nchw_tensor<Input_type>(input_shape.dims);
        Tensor y_tensor = Tensor::make_nchw_tensor<Input_type>(input_shape.dims);
        Tensor scale_tensor = Tensor::make_nchw_tensor<Intermediate_type>(channel_dims);
        Tensor bias_tensor = Tensor::make_nchw_tensor<Intermediate_type>(channel_dims);
        Tensor mean_tensor = Tensor::make_nchw_tensor<Intermediate_type>(channel_dims);
        Tensor variance_tensor = Tensor::make_nchw_tensor<Intermediate_type>(channel_dims);

        //todo, maybe randomize the data.

        std::unordered_map<int64_t, void*> variant_pack;
        variant_pack[x_tensor_attr->get_uid()] = x_tensor.memory().device_data<void>();
        variant_pack[mean_tensor_attr->get_uid()] = mean_tensor.memory().device_data<void>();
        variant_pack[inv_variance_tensor_attr->get_uid()]
            = variance_tensor.memory().device_data<void>();
        variant_pack[scale_tensor_attr->get_uid()] = scale_tensor.memory().device_data<void>();
        variant_pack[bias_tensor_attr->get_uid()] = bias_tensor.memory().device_data<void>();

        if(!y_tensor_attr->has_uid())
        {
            HIPDNN_LOG_INFO("y_tensor_attr does not have a UID, creating a new one.");
            y_tensor_attr->set_uid(6);
        }

        variant_pack[y_tensor_attr->get_uid()] = y_tensor.memory().device_data<void>();

        result = graph->execute(handle, variant_pack, nullptr);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    }

private:
    hipdnnHandle_t handle = nullptr;
    hipStream_t stream = nullptr;
    int device_id = 0;
};

// Test cases with various tensor shapes
TEST_F(BatchnormForwardInferenceIntegrationTest, SmallTensor_NCHW)
{
    TensorShape shape;
    shape.dims = {1, 3, 4, 4};
    shape.strides = {48, 16, 4, 1};
    shape.name = "SmallTensor_NCHW";

    RunBatchnormTest<float, float>(shape);
}

// TEST_F(BatchnormForwardInferenceIntegrationTest, MediumTensor_NCHW)
// {
//     TensorShape shape;
//     shape.dims = {2, 64, 32, 32};
//     shape.strides = {65536, 1024, 32, 1};
//     shape.name = "MediumTensor_NCHW";

//     RunBatchnormTest(shape);
// }

// TEST_F(BatchnormForwardInferenceIntegrationTest, LargeTensor_NCHW)
// {
//     TensorShape shape;
//     shape.dims = {4, 128, 56, 56};
//     shape.strides = {401408, 3136, 56, 1};
//     shape.name = "LargeTensor_NCHW";

//     RunBatchnormTest(shape);
// }

// TEST_F(BatchnormForwardInferenceIntegrationTest, SingleChannel_NCHW)
// {
//     TensorShape shape;
//     shape.dims = {1, 1, 8, 8};
//     shape.strides = {64, 64, 8, 1};
//     shape.name = "SingleChannel_NCHW";

//     RunBatchnormTest(shape);
// }

// TEST_F(BatchnormForwardInferenceIntegrationTest, ManyChannels_NCHW)
// {
//     TensorShape shape;
//     shape.dims = {1, 512, 7, 7};
//     shape.strides = {25088, 49, 7, 1};
//     shape.name = "ManyChannels_NCHW";

//     RunBatchnormTest(shape);
// }

// TEST_F(BatchnormForwardInferenceIntegrationTest, BatchSize8_NCHW)
// {
//     TensorShape shape;
//     shape.dims = {8, 32, 16, 16};
//     shape.strides = {8192, 256, 16, 1};
//     shape.name = "BatchSize8_NCHW";

//     RunBatchnormTest(shape);
// }

// TEST_F(BatchnormForwardInferenceIntegrationTest, Rectangle_NCHW)
// {
//     TensorShape shape;
//     shape.dims = {2, 16, 64, 32};
//     shape.strides = {32768, 2048, 32, 1};
//     shape.name = "Rectangle_NCHW";

//     RunBatchnormTest(shape);
// }

// // Parameterized test for easy shape variation
// class BatchnormParameterizedTest : public BatchnormForwardInferenceIntegrationTest,
//                                    public ::testing::WithParamInterface<TensorShape>
// {
// };

// TEST_P(BatchnormParameterizedTest, VariousShapes)
// {
//     RunBatchnormTest(GetParam());
// }

// INSTANTIATE_TEST_SUITE_P(
//     DifferentShapes,
//     BatchnormParameterizedTest,
//     ::testing::Values(TensorShape{{1, 3, 224, 224}, {150528, 50176, 224, 1}, "ImageNet_NCHW"},
//                       TensorShape{{16, 64, 14, 14}, {12544, 196, 14, 1}, "MiddleLayer_NCHW"},
//                       TensorShape{{32, 256, 7, 7}, {12544, 49, 7, 1}, "DeepLayer_NCHW"},
//                       TensorShape{{1, 1024, 1, 1}, {1024, 1, 1, 1}, "GlobalPool_NCHW"}));

// NOLINTEND