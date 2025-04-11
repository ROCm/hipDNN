// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "../attributes/graph_attributes.hpp"
#include "../attributes/tensor_attributes.hpp"
#include "../error.hpp"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
class INode // NOLINT
{
public:
    Graph_attributes graph_attributes;
    INode(Graph_attributes attributes)
        : graph_attributes(std::move(attributes))
    {
    }
    virtual ~INode() = default;

private:
    virtual error_t pre_validate_node() const
    {
        return {};
    }
    virtual error_t infer_properties_node()
    {
        return {};
    }
    virtual error_t post_validate_node() const
    {
        return {};
    }
    virtual error_t populate_hipdnn_tensor_ids(
        std::unordered_map<int64_t, std::shared_ptr<Tensor_attributes>>& tensor_lookup,
        int64_t&                                                         current_tensor_id,
        std::unordered_set<int64_t>&                                     used_ids) const
    {
        return {};
    }
    virtual void gather_hipdnn_tensor_ids(std::unordered_set<int64_t>& used_ids) const {

    };

protected:
    std::vector<std::shared_ptr<INode>> _sub_nodes;

    error_t validate_subtree()
    {
        CHECK_HIPDNN_ERROR(pre_validate_node());
        CHECK_HIPDNN_ERROR(infer_properties_node());
        for(const auto& node : _sub_nodes)
        {
            CHECK_HIPDNN_ERROR(node->validate_subtree());
        }
        CHECK_HIPDNN_ERROR(post_validate_node());
        return {};
    }

    void gather_hipdnn_tensor_ids_subtree(std::unordered_set<int64_t>& used_ids) const
    {
        gather_hipdnn_tensor_ids(used_ids);
        for(const auto& node : _sub_nodes)
        {
            node->gather_hipdnn_tensor_ids_subtree(used_ids);
        }
    }

    error_t populate_hipdnn_tensor_ids_subtree(
        std::unordered_map<int64_t, std::shared_ptr<Tensor_attributes>>& tensor_lookup,
        int64_t&                                                         current_tensor_id,
        std::unordered_set<int64_t>&                                     used_ids)
    {
        CHECK_HIPDNN_ERROR(populate_hipdnn_tensor_ids(tensor_lookup, current_tensor_id, used_ids));
        for(const auto& node : _sub_nodes)
        {
            CHECK_HIPDNN_ERROR(
                node->populate_hipdnn_tensor_ids(tensor_lookup, current_tensor_id, used_ids));
        }
        return {};
    }
};

// Any class extending NodeCRTP must have an attributes member with an inputs & outputs map.
// The map needs to have TensorAttributes as the value.
// NodeCRTP uses this to gather tensor uids, and populate unset ones.
template <typename DerivedT>
class NodeCRTP : public INode // NOLINT
{
private:
    DerivedT& self()
    {
        return static_cast<DerivedT&>(*this);
    }
    const DerivedT& self() const
    {
        return static_cast<const DerivedT&>(*this);
    }
    void gather_hipdnn_tensor_ids(std::unordered_set<int64_t>& used_ids) const override
    {
        for(auto& [_, tensor] : self().attributes.inputs)
        {
            if(tensor && tensor->has_uid())
            {
                used_ids.insert(tensor->get_uid());
            }
        }

        for(auto& [_, tensor] : self().attributes.outputs)
        {
            if(tensor && tensor->has_uid())
            {
                used_ids.insert(tensor->get_uid());
            }
        }
    }

    static int64_t get_unused_tensor_uid(int64_t&                     current_tensor_id,
                                         std::unordered_set<int64_t>& used_ids)
    {
        while(used_ids.find(current_tensor_id) != used_ids.end())
        {
            ++current_tensor_id;
        }
        used_ids.insert(current_tensor_id);
        return current_tensor_id++;
    }

    error_t populate_hipdnn_tensor_ids(
        std::unordered_map<int64_t, std::shared_ptr<Tensor_attributes>>& tensor_lookup,
        int64_t&                                                         current_tensor_id,
        std::unordered_set<int64_t>&                                     used_ids) const override
    {
        for(auto& [_, tensor] : self().attributes.inputs)
        {
            if(tensor && !tensor->has_uid())
            {
                tensor->set_uid(get_unused_tensor_uid(current_tensor_id, used_ids));
            }

            tensor_lookup[tensor->get_uid()] = tensor;
        }

        for(auto& [_, tensor] : self().attributes.outputs)
        {
            if(tensor && !tensor->has_uid())
            {
                tensor->set_uid(get_unused_tensor_uid(current_tensor_id, used_ids));
            }
            tensor_lookup[tensor->get_uid()] = tensor;
        }

        return {};
    }

protected:
    using INode::INode;
};
}
}
