// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "node.hpp"
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/graph_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_frontend::graph
{
class BatchnormInferenceNode : public NodeCRTP<BatchnormInferenceNode> //NOLINT
{
public:
    Batchnorm_inference_attributes attributes;

    BatchnormInferenceNode(Batchnorm_inference_attributes&& batchnorm_attrs,
                           const Graph_attributes&          graph_attrs)
        : NodeCRTP(graph_attrs)
        , attributes(std::move(batchnorm_attrs))
    {
    }

    error_t pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing x for pre-validation"};
        }
        if(!attributes.get_scale())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing scale for pre-validation"};
        }
        if(!attributes.get_bias())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing bias for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing y for pre-validation"};
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
                    "BatchnormInferenceNode missing x for setting properties"};
        }

        if(!y)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing y for setting properties"};
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

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.name.c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormInferenceAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
}
