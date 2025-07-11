/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include "miopen_batchnorm_solver.hpp"
#include <hipdnn_sdk/plugin/plugin_flatbuffer_type_helpers.hpp>
#include <miopen/miopen.h>

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>

namespace miopen_legacy_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t miopen_batchnorm_mode = miopenBNSpatial;

bool Miopen_batchnorm_solver::is_applicable(const hipdnn_plugin::Graph_interface& op_graph) const
{

    if(op_graph.node_count() != 1)
    {
        HIPDNN_LOG_INFO(
            "Batchnorm solver is applicable only for single node graphs. Graph has {} nodes",
            op_graph.node_count());
        return false;
    }

    if(!op_graph.has_only_supported_attributes(std::set<hipdnn_sdk::data_objects::NodeAttributes>{
           hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes}))
    {
        HIPDNN_LOG_INFO("Batchnorm solver is not applicable for this graph");
        return false;
    }

    return true;
}

size_t Miopen_batchnorm_solver::get_workspace_size(
    const hipdnnEnginePluginHandle& handle, const hipdnn_plugin::Graph_interface& op_graph) const
{
    //batchnorm solver does not require workspace size
    return 0u;
}

void Miopen_batchnorm_solver::execute_graph(
    const hipdnnEnginePluginHandle& handle,
    const hipdnnEnginePluginExecutionContext& execution_context,
    const hipdnnPluginDeviceBuffer_t* device_buffers,
    uint32_t num_device_buffers,
    void* workspace)
{
    const auto& node = execution_context.graph().get_node(0);

    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes:
        execute_batchnorm_fwd_inference(handle,
                                        execution_context.graph(),
                                        *node.attributes_as_BatchnormInferenceAttributes(),
                                        device_buffers,
                                        num_device_buffers);
        break;
    default:
        throw hipdnn_plugin::Hipdnn_plugin_exception(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for batchnorm solver: "
                + std::string(hipdnn_sdk::data_objects::to_string(node.attributes_type())));
    }
}

struct MiOpenTensorAndDeviceBufferPair
{
    miopenTensorDescriptor_t tensor_desc;
    hipdnnPluginDeviceBuffer_t device_buffer;
};

miopenDataType_t
    tensor_data_type_to_miopen_data_type(const hipdnn_sdk::data_objects::DataType& data_type)
{
    switch(data_type)
    {
    case hipdnn_sdk::data_objects::DataType_FLOAT:
        return miopenFloat;
    case hipdnn_sdk::data_objects::DataType_HALF:
        return miopenHalf;
    case hipdnn_sdk::data_objects::DataType_BFLOAT16:
        return miopenBFloat16;
    default:
        throw hipdnn_plugin::Hipdnn_plugin_exception(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported data type for MIOpen: "
                + std::string(hipdnn_sdk::data_objects::to_string(data_type)));
    }
}

void create_tensor_and_device_buffer_pair(const hipdnn_sdk::data_objects::TensorAttributes& tensor,
                                          const hipdnnPluginDeviceBuffer_t* device_buffers,
                                          uint32_t num_device_buffers,
                                          MiOpenTensorAndDeviceBufferPair& pair)
{
    //todo deal with status's

    miopenCreateTensorDescriptor(&pair.tensor_desc);

    std::vector<int> dims(tensor.dims()->begin(), tensor.dims()->end());
    std::vector<int> strides(tensor.strides()->begin(), tensor.strides()->end());
    miopenSetTensorDescriptor(pair.tensor_desc,
                              tensor_data_type_to_miopen_data_type(tensor.data_type()),
                              dims.size(),
                              dims.data(),
                              strides.data());

    for(uint32_t i = 0; i < num_device_buffers; i++)
    {
        if(tensor.uid() == device_buffers[i].uid)
        {
            pair.device_buffer = device_buffers[i];
            break;
        }
    }
}

void Miopen_batchnorm_solver::execute_batchnorm_fwd_inference(
    const hipdnnEnginePluginHandle& handle,
    hipdnn_plugin::Graph_interface& op_graph,
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const hipdnnPluginDeviceBuffer_t* device_buffers,
    uint32_t num_device_buffers)
{
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);
    double epsilon = 1e-3; // taken from bn driver, todo, figure out better way

    const auto& tensor_map = op_graph.get_tensor_map();

    MiOpenTensorAndDeviceBufferPair xDesc;
    MiOpenTensorAndDeviceBufferPair yDesc;

    MiOpenTensorAndDeviceBufferPair scaleDesc;
    MiOpenTensorAndDeviceBufferPair biasDesc;
    MiOpenTensorAndDeviceBufferPair estMeanDesc;
    MiOpenTensorAndDeviceBufferPair estVarianceDesc;

    create_tensor_and_device_buffer_pair(
        *tensor_map.at(attributes.x()), device_buffers, num_device_buffers, xDesc);

    create_tensor_and_device_buffer_pair(
        *tensor_map.at(attributes.y()), device_buffers, num_device_buffers, yDesc);

    create_tensor_and_device_buffer_pair(
        *tensor_map.at(attributes.scale()), device_buffers, num_device_buffers, scaleDesc);

    create_tensor_and_device_buffer_pair(
        *tensor_map.at(attributes.bias()), device_buffers, num_device_buffers, biasDesc);

    //todo, below are optional, need to check if they exist and only set then
    create_tensor_and_device_buffer_pair(
        *tensor_map.at(attributes.mean().value()), device_buffers, num_device_buffers, estMeanDesc);

    create_tensor_and_device_buffer_pair(*tensor_map.at(attributes.inv_variance().value()),
                                         device_buffers,
                                         num_device_buffers,
                                         estVarianceDesc);

    auto miopen_status
        = miopenBatchNormalizationForwardInference_V2(handle.miopen_handle,
                                                      miopen_batchnorm_mode,
                                                      &alpha,
                                                      &beta,
                                                      xDesc.tensor_desc,
                                                      xDesc.device_buffer.ptr,
                                                      yDesc.tensor_desc,
                                                      yDesc.device_buffer.ptr,
                                                      scaleDesc.tensor_desc,
                                                      biasDesc.tensor_desc,
                                                      estMeanDesc.tensor_desc,
                                                      estVarianceDesc.tensor_desc,
                                                      scaleDesc.device_buffer.ptr,
                                                      biasDesc.device_buffer.ptr,
                                                      estMeanDesc.device_buffer.ptr,
                                                      estVarianceDesc.device_buffer.ptr,
                                                      epsilon);

    HIPDNN_LOG_INFO("MIOpen batchnorm forward inference status: {}",
                    miopenGetErrorString(miopen_status));
}

}
