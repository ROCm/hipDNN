// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_frontend::graph
{
class PointwiseNode : public BaseNode<PointwiseNode>
{
public:
    PointwiseAttributes attributes;

    PointwiseNode(PointwiseAttributes&& batchnormAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(batchnormAttrs))
    {
    }

    error_t pre_validate_node() const override
    {
        if(!attributes.get_input_0())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing IN_0 for pre-validation"};
        }
        if(!attributes.get_output_0())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing OUT_0 for pre-validation"};
        }
        if(attributes.get_mode() == PointwiseMode_t::NOT_SET)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing operation for pre-validation"};
        }

        return {};
    }

    error_t infer_properties_node() override
    {
        auto in0 = attributes.get_input_0();
        if(!in0)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing input for setting properties"};
        }

        auto out = attributes.get_output_0();
        if(!out)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing output for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        if(out->get_dim().empty())
        {
            std::vector<std::vector<int64_t>> inputShapes;
            std::vector<std::shared_ptr<TensorAttributes>> allInputs
                = {in0, attributes.get_input_1(), attributes.get_input_2()};

            for(const auto& tensor : allInputs)
            {
                if(tensor)
                {
                    inputShapes.push_back(tensor->get_dim());
                }
            }

            auto outputDims = out->get_dim();
            HIPDNN_CHECK_ERROR(findCommonShape(inputShapes, outputDims));
            out->set_dim(outputDims);
        }

        // Note: Strides will only be set if there is an input tensor with the same dims currently.
        //       This could fail if the input tensors were something like (1, 1, 2) and (2, 1, 1).
        if(out->get_stride().empty())
        {
            std::vector<std::shared_ptr<TensorAttributes>> allInputs
                = {in0, attributes.get_input_1(), attributes.get_input_2()};

            for(const auto& tensor : allInputs)
            {
                if(tensor && tensor->get_dim() == out->get_dim())
                {
                    out->set_stride(tensor->get_stride());
                    break;
                }
            }

            if(out->get_stride().empty())
            {
                return {error_code_t::ATTRIBUTE_NOT_SET, "PointwiseNode output missing stride"};
            }
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_PointwiseAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
}
