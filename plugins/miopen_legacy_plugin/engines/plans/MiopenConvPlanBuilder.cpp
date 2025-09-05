/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <algorithm>
#include <limits>
#include <string>

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <miopen/miopen.h>

#include "engines/plans/MiopenConvFwdPlan.hpp"
#include "MiopenConvDescriptor.hpp"
#include "MiopenConvPlanBuilder.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

namespace
{

constexpr size_t MIN_SUPPORTED_TENSOR_DIMS = 4;
constexpr size_t MAX_SUPPORTED_TENSOR_DIMS = 5;

std::string getNodeName(const hipdnn_sdk::data_objects::Node& node)
{
    return node.name() != nullptr ? node.name()->str() : "";
}

bool isApplicableFwd(const HipdnnEnginePluginHandle& handle, const hipdnn_plugin::IGraph& opGraph)
{
    const auto& node = opGraph.getNode(0);

    const auto* attr = node.attributes_as_ConvolutionFwdAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to ConvolutionFwdAttributes for node: "
                + getNodeName(node));
    }

    // Check convolution mode

    const auto convMode = attr->conv_mode();
    if(convMode != hipdnn_sdk::data_objects::ConvMode::ConvMode_CROSS_CORRELATION)
    {
        HIPDNN_LOG_INFO("Convolution plan builder supports only CROSS_CORRELATION");
        return false;
    }

    // Check tensor attributes

    const auto& tensorMap = opGraph.getTensorMap();
    const auto& tensorAttrX = miopen_utils::findTensorAttributes(tensorMap, attr->x_tensor_uid());
    const auto& tensorAttrW = miopen_utils::findTensorAttributes(tensorMap, attr->w_tensor_uid());
    const auto& tensorAttrY = miopen_utils::findTensorAttributes(tensorMap, attr->y_tensor_uid());

    if(tensorAttrX.dims()->size() < MIN_SUPPORTED_TENSOR_DIMS || tensorAttrX.dims()->size() > MAX_SUPPORTED_TENSOR_DIMS)
    {
        HIPDNN_LOG_INFO("Convolution plan builder supports only tensors with " + 
                        std::to_string(MIN_SUPPORTED_TENSOR_DIMS) + " to " +
                        std::to_string(MAX_SUPPORTED_TENSOR_DIMS) + " dimensions");
        return false;
    }

    if(tensorAttrY.dims()->size() != tensorAttrX.dims()->size() ||
       tensorAttrW.dims()->size() != tensorAttrX.dims()->size())
    {
        HIPDNN_LOG_WARN("Convolution plan builder requires all tensors to have the same number of dimensions");
        return false;
    }

    // Check convolution attributes

    const auto prePadding = attr->pre_padding();
    const auto postPadding = attr->post_padding();
    const auto stride = attr->stride();
    const auto dilation = attr->dilation();

    const auto spatialDimCount = miopen_utils::getSpatialDimCount(tensorAttrX);

    auto checkVectorSize = [&](const auto* vec, const char* name) {
        if (vec != nullptr && vec->size() != spatialDimCount) {
            HIPDNN_LOG_WARN("Convolution plan builder: " + std::string(name) + " size does not match spatial dimension count");
            return false;
        }
        return true;
    };

    if (!checkVectorSize(prePadding, "prePadding") ||
        !checkVectorSize(postPadding, "postPadding") ||
        !checkVectorSize(stride, "stride") ||
        !checkVectorSize(dilation, "dilation")) {
        return false;
    }

    // Check padding symmetry

    if((prePadding == nullptr) != (postPadding == nullptr))
    {
        HIPDNN_LOG_INFO("Convolution plan builder requires both prePadding and postPadding to be set or both to be null");
        return false;
    }

    if(prePadding != nullptr && postPadding != nullptr)
    {
        // flatbuffers::Vector does not have comparison operators
        if (!std::equal(prePadding->cbegin(), prePadding->cend(), postPadding->cbegin())) 
        {
            HIPDNN_LOG_INFO("Convolution plan builder supports only symmetric padding");
            return false;
        }
    }

    // integer overflow + correctness checks

    auto checkVectorMinValue = [](const auto* vec, const char* name, int64_t minValue) {
        if (vec != nullptr) {
            if (std::any_of(vec->cbegin(), vec->cend(), [&](auto v) { return v < minValue; })) {
                HIPDNN_LOG_WARN("Convolution plan builder: " + std::string(name) + " has value less than " + std::to_string(minValue));
                return false;
            }
        }
        return true;
    };

    if(!checkVectorMinValue(prePadding, "prePadding", 0) ||
       !checkVectorMinValue(stride, "stride", 1) ||
       !checkVectorMinValue(dilation, "dilation", 1))
    {
        return false;
    }

    auto checkVectorIntegerOverflow = [](const auto* vec, const char* name) {
        if (vec != nullptr) {
            if (std::any_of(vec->cbegin(), vec->cend(), [](auto v) { return v > static_cast<int64_t>(std::numeric_limits<int>::max()); })) {
                HIPDNN_LOG_INFO("Convolution plan builder: " + std::string(name) + " has value greater than INT_MAX");
                return false;
            }
        }
        return true;
    };

    if(!checkVectorIntegerOverflow(prePadding, "prePadding") ||
       !checkVectorIntegerOverflow(stride, "stride") ||
       !checkVectorIntegerOverflow(dilation, "dilation"))
    {
        return false;
    }

    // Create MIOpen tensor descriptors
    const MiopenTensor tensorX(tensorAttrX);
    const MiopenTensor tensorW(tensorAttrW);
    const MiopenTensor tensorY(tensorAttrY);

    // Create MIOpen convolution descriptor
    const MiopenConvDescriptor convDesc(spatialDimCount, *attr);

    size_t solutionCount;
    THROW_ON_MIOPEN_FAILURE(miopenConvolutionForwardGetSolutionCount(handle.miopenHandle,
                                                                     tensorW.tensorDescriptor(),
                                                                     tensorX.tensorDescriptor(),
                                                                     convDesc.convDescriptor(),
                                                                     tensorY.tensorDescriptor(),
                                                                     &solutionCount));

    return solutionCount != 0;
}

size_t getWorkspaceSizeFwd(const HipdnnEnginePluginHandle& handle, const hipdnn_plugin::IGraph& opGraph)
{
    const auto& node = opGraph.getNode(0);

    const auto* attr = node.attributes_as_ConvolutionFwdAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to ConvolutionFwdAttributes for node: "
                + getNodeName(node));
    }

    ConvFwdParams params(*attr, opGraph.getTensorMap());
    size_t workSpaceSize;
    THROW_ON_MIOPEN_FAILURE(miopenConvolutionForwardGetWorkSpaceSize(handle.miopenHandle,
                                                                     params.w().tensorDescriptor(),
                                                                     params.x().tensorDescriptor(),
                                                                     params.conv().convDescriptor(),
                                                                     params.y().tensorDescriptor(),
                                                                     &workSpaceSize));

    return workSpaceSize; 
}

void buildPlanFwd(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& node = opGraph.getNode(0);

    const auto* attr = node.attributes_as_ConvolutionFwdAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to ConvolutionFwdAttributes for node: "
                + getNodeName(node));
    }

    auto params = std::make_unique<ConvFwdParams>(*attr, opGraph.getTensorMap());
    auto plan = std::make_unique<ConvFwdPlan>(handle, std::move(params));
    executionContext.setPlan(std::move(plan));
}

} // namespace

bool MiopenConvPlanBuilder::isApplicable(const HipdnnEnginePluginHandle& handle, const hipdnn_plugin::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_LOG_INFO(
            "Convolution plan builder is applicable only for single node graphs. Graph has {} nodes",
            opGraph.nodeCount());
        return false;
    }

    const auto& node = opGraph.getNode(0);
    bool ret = false;

    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes_ConvolutionFwdAttributes:
        ret = isApplicableFwd(handle, opGraph);
        break;
    default:
        break;
    }

    if(!ret)
    {
        HIPDNN_LOG_INFO("Convolution plan builder is not applicable for this graph");
    }
    return ret;
}

size_t MiopenConvPlanBuilder::getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                                    const hipdnn_plugin::IGraph& opGraph) const
{
    const auto& node = opGraph.getNode(0);

    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes_ConvolutionFwdAttributes:
        return getWorkspaceSizeFwd(handle, opGraph);
    default:
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(hipdnn_sdk::data_objects::toString(node.attributes_type())));
    }
}

void MiopenConvPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    const auto& node = opGraph.getNode(0);

    std::string nodeName = getNodeName(node);
    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes_ConvolutionFwdAttributes:
        HIPDNN_LOG_INFO("Building convolution fwd plan for node: {}", nodeName);
        buildPlanFwd(handle, opGraph, executionContext);
        break;
    default:
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(hipdnn_sdk::data_objects::toString(node.attributes_type())));
    }
}

} // namespace miopen_legacy_plugin
