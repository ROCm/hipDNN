// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "node.hpp"
#include <hipdnn_frontend/attributes/batchnorm_attributes.hpp>
#include <hipdnn_frontend/attributes/graph_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/utilities.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_frontend::graph
{
class BatchNormNode : public NodeCRTP<BatchNormNode> //NOLINT
{
public:
    Batchnorm_attributes attributes;

    BatchNormNode(Batchnorm_attributes&& batchnorm_attrs, const Graph_attributes& graph_attrs)
        : NodeCRTP(graph_attrs)
        , attributes(std::move(batchnorm_attrs))
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

        CHECK_HIPDNN_ERROR(attributes.fill_from_graph_attributes(graph_attributes));

        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty())
        {
            y->set_stride(x->get_stride());
        }

        auto infer_c_tensor = [&](std::shared_ptr<Tensor_attributes>& tensorToInfer) {
            if(tensorToInfer->get_dim().empty())
            {
                std::vector<int64_t> tensor_dims(x->get_dim().size(), 1);
                tensor_dims[1] = x->get_dim()[1];
                tensorToInfer->set_dim(tensor_dims);
            }

            if(tensorToInfer->get_stride().empty())
            {
                auto stride_order = stride_order_nhwc(tensorToInfer->get_dim().size());
                tensorToInfer->set_stride(generate_strides(tensorToInfer->get_dim(), stride_order));
            }
        };

        auto mean    = attributes.get_mean();
        auto inv_var = attributes.get_inv_variance();

        if(mean)
        {
            infer_c_tensor(mean);
        }

        if(inv_var)
        {
            infer_c_tensor(inv_var);
        }

        auto prev_running_mean = attributes.get_prev_running_mean();
        auto prev_running_var  = attributes.get_prev_running_variance();

        auto next_running_mean = attributes.get_next_running_mean();
        auto next_running_var  = attributes.get_next_running_variance();

        if(prev_running_mean && prev_running_var && next_running_mean && next_running_var)
        {
            infer_c_tensor(next_running_mean);
            infer_c_tensor(next_running_var);
        }

        return {};
    }

    void gather_hipdnn_tensor_ids(std::unordered_set<int64_t>& used_ids) const override
    {
        NodeCRTP<BatchNormNode>::gather_hipdnn_tensor_ids(used_ids);

        for(auto& tensor : attributes.peer_stats)
        {
            if(tensor && tensor->has_uid())
            {
                used_ids.insert(tensor->get_uid());
            }
        }
    }

    error_t populate_hipdnn_tensor_ids(
        std::unordered_map<int64_t, std::shared_ptr<Tensor_attributes>>& tensor_lookup,
        int64_t&                                                         current_tensor_id,
        std::unordered_set<int64_t>&                                     used_ids) const override
    {
        NodeCRTP<BatchNormNode>::populate_hipdnn_tensor_ids(
            tensor_lookup, current_tensor_id, used_ids);

        for(auto& tensor : attributes.peer_stats)
        {
            if(tensor && !tensor->has_uid())
            {
                tensor->set_uid(get_unused_tensor_uid(current_tensor_id, used_ids));
            }

            tensor_lookup[tensor->get_uid()] = tensor;
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.name.c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormAttributes,
            attributes.pack_attributes(builder).Union());
    }
};

typedef BatchNormNode BatchnormNode;
}