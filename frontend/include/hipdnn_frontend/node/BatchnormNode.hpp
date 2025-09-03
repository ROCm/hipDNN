// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/ShapeUtils.hpp>

namespace hipdnn_frontend::graph
{
class BatchNormNode : public NodeCRTP<BatchNormNode> //NOLINT
{
public:
    BatchnormAttributes attributes;

    BatchNormNode(BatchnormAttributes&& batchnormAttrs, const GraphAttributes& graphAttrs)
        : NodeCRTP(graphAttrs)
        , attributes(std::move(batchnormAttrs))
    {
    }

    error_t pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET, "BatchnormNode missing x for pre-validation"};
        }
        if(!attributes.get_scale())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormNode missing scale for pre-validation"};
        }
        if(!attributes.get_bias())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormNode missing bias for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET, "BatchnormNode missing y for pre-validation"};
        }
        if(!attributes.get_epsilon())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormNode missing epsilon for pre-validation"};
        }

        return {};
    }

    error_t infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(!x)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormNode missing x for setting properties"};
        }

        if(!y)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormNode missing y for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_graph_attributes(graph_attributes));

        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty())
        {
            y->set_stride(x->get_stride());
        }

        auto inferCTensor = [&](std::shared_ptr<TensorAttributes>& tensorToInfer) {
            if(tensorToInfer->get_dim().empty())
            {
                std::vector<int64_t> tensorDims(x->get_dim().size(), 1);
                tensorDims[1] = x->get_dim()[1];
                tensorToInfer->set_dim(tensorDims);
            }

            if(tensorToInfer->get_stride().empty())
            {
                auto strideOrder
                    = hipdnn_sdk::utilities::strideOrderNhwc(tensorToInfer->get_dim().size());
                tensorToInfer->set_stride(
                    hipdnn_sdk::utilities::generateStrides(tensorToInfer->get_dim(), strideOrder));
            }
        };

        auto mean = attributes.get_mean();
        auto invVar = attributes.get_inv_variance();

        if(mean)
        {
            inferCTensor(mean);
        }

        if(invVar)
        {
            inferCTensor(invVar);
        }

        auto prevRunningMean = attributes.get_prev_running_mean();
        auto prevRunningVar = attributes.get_prev_running_variance();

        auto nextRunningMean = attributes.get_next_running_mean();
        auto nextRunningVar = attributes.get_next_running_variance();

        if(prevRunningMean && prevRunningVar && nextRunningMean && nextRunningVar)
        {
            inferCTensor(nextRunningMean);
            inferCTensor(nextRunningVar);
        }

        return {};
    }

    void gather_hipdnn_tensor_ids(std::unordered_set<int64_t>& usedIds) const override
    {
        NodeCRTP<BatchNormNode>::gather_hipdnn_tensor_ids(usedIds);

        for(auto& tensor : attributes.peer_stats)
        {
            if(tensor && tensor->has_uid())
            {
                usedIds.insert(tensor->get_uid());
            }
        }
    }

    error_t populate_hipdnn_tensor_ids(
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorLookup,
        int64_t& currentTensorId,
        std::unordered_set<int64_t>& usedIds) const override
    {
        NodeCRTP<BatchNormNode>::populate_hipdnn_tensor_ids(tensorLookup, currentTensorId, usedIds);

        for(auto& tensor : attributes.peer_stats)
        {
            if(tensor && !tensor->has_uid())
            {
                tensor->set_uid(get_unused_tensor_uid(currentTensorId, usedIds));
            }

            tensorLookup[tensor->get_uid()] = tensor;
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormAttributes,
            attributes.pack_attributes(builder).Union());
    }
};

typedef BatchNormNode BatchnormNode;
}
