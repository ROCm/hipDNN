// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "node.hpp"
#include <hipdnn_frontend/attributes/batchnorm_backward_attributes.hpp>
#include <hipdnn_frontend/attributes/graph_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/utilities.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_frontend::graph
{

class DBNNode : public NodeCRTP<DBNNode> //NOLINT
{
public:
    Batchnorm_backward_attributes attributes;

    DBNNode(Batchnorm_backward_attributes&& batchnorm_attrs, const Graph_attributes& graph_attrs)
        : NodeCRTP(graph_attrs)
        , attributes(std::move(batchnorm_attrs))
    {
    }

    error_t pre_validate_node() const override
    {
        if(!attributes.get_dy())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dy for pre-validation"};
        }
        if(!attributes.get_x())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing x for pre-validation"};
        }
        if(!attributes.get_scale())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing scale for pre-validation"};
        }
        if(!attributes.get_dx())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dx for pre-validation"};
        }
        if(!attributes.get_dscale())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dscale for pre-validation"};
        }
        if(!attributes.get_dbias())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dbias for pre-validation"};
        }

        return {};
    }

    error_t infer_properties_node() override
    {
        auto x      = attributes.get_x();
        auto dx     = attributes.get_dx();
        auto dscale = attributes.get_dscale();
        auto dbias  = attributes.get_dbias();

        if(!x)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing x for setting properties"};
        }
        if(!dx)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dx for setting properties"};
        }
        if(!dscale)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dscale for setting properties"};
        }
        if(!dbias)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dbias for setting properties"};
        }

        CHECK_HIPDNN_ERROR(attributes.fill_from_graph_attributes(graph_attributes));

        if(dx->get_dim().empty())
        {
            dx->set_dim(x->get_dim());
        }

        if(dx->get_stride().empty())
        {
            dx->set_stride(x->get_stride());
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

        infer_c_tensor(dscale);
        infer_c_tensor(dbias);

        return {};
    }

    void gather_hipdnn_tensor_ids(std::unordered_set<int64_t>& used_ids) const override
    {
        NodeCRTP<DBNNode>::gather_hipdnn_tensor_ids(used_ids);

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
        NodeCRTP<DBNNode>::populate_hipdnn_tensor_ids(tensor_lookup, current_tensor_id, used_ids);

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
            hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormBackwardAttributes,
            attributes.pack_attributes(builder).Union());
    }
};

typedef DBNNode BatchnormBackwardNode;
}