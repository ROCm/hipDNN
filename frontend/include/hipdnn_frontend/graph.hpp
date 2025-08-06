// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "flatbuffers/detached_buffer.h"
#include <hipdnn_frontend/attributes/batchnorm_attributes.hpp>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/backend/backend_wrapper.hpp>
#include <hipdnn_frontend/backend/scoped_hipdnn_backend_descriptor.hpp>
#include <hipdnn_frontend/node/batchnorm_backward_node.hpp>
#include <hipdnn_frontend/node/batchnorm_inference_node.hpp>
#include <hipdnn_frontend/node/batchnorm_node.hpp>
#include <hipdnn_frontend/node/node.hpp>
#include <hipdnn_frontend/node/pointwise_node.hpp>

namespace hipdnn_frontend
{
namespace graph
{
// When an error occurs, get the backend error string and append it to the error_message.
#define RETURN_ON_BACKEND_FAILURE(backend_status, error_message)                         \
    if((backend_status) != HIPDNN_STATUS_SUCCESS)                                        \
    {                                                                                    \
        std::array<char, 256> backend_err_msg{};                                         \
        hipdnn_frontend::hipdnn_backend().get_last_error_string(backend_err_msg.data(),  \
                                                                backend_err_msg.size()); \
        std::string full_error_msg                                                       \
            = std::string(error_message) + " Backend error: " + backend_err_msg.data();  \
        return error_t(error_code_t::HIPDNN_BACKEND_ERROR, full_error_msg);              \
    }

class Graph : public INode
{
private:
    std::unique_ptr<Scoped_hipdnn_backend_descriptor> _graph_desc;
    std::unique_ptr<Scoped_hipdnn_backend_descriptor> _engine_heuristic_desc;
    std::unique_ptr<Scoped_hipdnn_backend_descriptor> _engine_config_desc;
    std::unique_ptr<Scoped_hipdnn_backend_descriptor> _execution_plan_desc;

    static std::shared_ptr<Tensor_attributes> output_tensor(const std::string& name)
    {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_name(name).set_is_virtual(true);
        return tensor;
    }

    error_t initialize_heuristic_descriptor(std::vector<HeurMode_t> const& modes)
    {
        _engine_heuristic_desc = std::make_unique<Scoped_hipdnn_backend_descriptor>(
            HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);

        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_set_attribute(_engine_heuristic_desc->get(),
                                                   HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_graph_desc->get()),
            "Failed to set operation graph on the engine heuristic descriptor.");

        // TODO
        // Currently we only handle the first mode in the vector.  Once we add heuristics we will need
        // to handle using all modes that are passed in.  We currently only have 1 mode so there
        // is only 1 possibility.
        std::vector<hipdnnBackendHeurMode_t> backend_modes;
        backend_modes.reserve(modes.size());
        for(const auto& mode : modes)
        {
            backend_modes.push_back(to_backend_type(mode));
        }

        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_set_attribute(_engine_heuristic_desc->get(),
                                                   HIPDNN_ATTR_ENGINEHEUR_MODE,
                                                   HIPDNN_TYPE_HEUR_MODE,
                                                   1,
                                                   backend_modes.data()),
            "Failed to set mode on the engine heuristic descriptor.");

        RETURN_ON_BACKEND_FAILURE(hipdnn_backend().backend_finalize(_engine_heuristic_desc->get()),
                                  "Failed to finalize engine heuristic descriptor");

        return {error_code_t::OK, ""};
    }

    error_t initialize_engine_config()
    {
        int64_t available_engine_count = 0;
        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_get_attribute(_engine_heuristic_desc->get(),
                                                   HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   0,
                                                   &available_engine_count,
                                                   nullptr),
            "Failed to get attribue from the engine heuristic descriptor.");

        if(available_engine_count == 0)
        {
            return {error_code_t::HIPDNN_BACKEND_ERROR,
                    "No engine configurations available for the graph."};
        }

        int required_count = 1;
        std::vector<std::unique_ptr<Scoped_hipdnn_backend_descriptor>> engine_configs;
        std::vector<hipdnnBackendDescriptor_t> engine_configs_shallow;
        for(size_t i = 0; std::cmp_less(i, required_count); ++i)
        {
            auto engine_cfg_desc = std::make_unique<Scoped_hipdnn_backend_descriptor>(
                HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);

            if(engine_cfg_desc == nullptr || !engine_cfg_desc->valid())
            {
                return {error_code_t::HIPDNN_BACKEND_ERROR,
                        "Failed to create engine configuration descriptor."};
            }
            engine_configs.push_back(std::move(engine_cfg_desc));
            engine_configs_shallow.push_back(engine_configs.back()->get());
        }

        int64_t count = 0;
        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_get_attribute(
                _engine_heuristic_desc->get(),
                HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                static_cast<int64_t>(engine_configs_shallow.size()),
                &count,
                engine_configs_shallow.data()),
            "Failed to get engine configurations from the heuristic descriptor.");

        if(count == 0)
        {
            return {error_code_t::HIPDNN_BACKEND_ERROR,
                    "No engine configurations retrieved from the heuristic desc."};
        }

        //TODO
        // Add filtering and logic to select the best engine configuration that meets the requirements.
        _engine_config_desc = std::move(engine_configs[0]);
        engine_configs.erase(engine_configs.begin(), engine_configs.begin() + 1);

        return {error_code_t::OK, ""};
    }

public:
    Graph()
        : INode(Graph_attributes{})
    {
    }

    error_t validate()
    {
        return validate_subtree();
    }

    error_t build_operation_graph(hipdnnHandle_t handle)
    {
        std::unordered_set<int64_t> used_tensor_uids;
        gather_hipdnn_tensor_ids_subtree(used_tensor_uids);

        std::unordered_map<int64_t, std::shared_ptr<Tensor_attributes>> tensor_lookup;
        int64_t current_tensor_id = 0;

        populate_hipdnn_tensor_ids_subtree(tensor_lookup, current_tensor_id, used_tensor_uids);
        flatbuffers::FlatBufferBuilder builder;

        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
            tensor_attributes;
        for(auto& [_, tensor] : tensor_lookup)
        {
            if(tensor)
            {
                tensor_attributes.emplace_back(tensor->pack_attributes(builder));
            }
        }

        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
        for(auto& node : _sub_nodes)
        {
            if(node)
            {
                nodes.emplace_back(node->pack_node(builder));
            }
        }
        auto graph = hipdnn_sdk::data_objects::CreateGraphDirect(
            builder,
            graph_attributes.get_name().c_str(),
            to_sdk_type(graph_attributes.get_compute_data_type()),
            to_sdk_type(graph_attributes.get_intermediate_data_type()),
            to_sdk_type(graph_attributes.get_io_data_type()),
            &tensor_attributes,
            &nodes);

        builder.Finish(graph);
        auto serialized_graph = builder.Release();
        _graph_desc = std::make_unique<Scoped_hipdnn_backend_descriptor>(serialized_graph.data(),
                                                                         serialized_graph.size());

        if(!_graph_desc->valid())
        {
            return {error_code_t::HIPDNN_BACKEND_ERROR,
                    "Failed to create backend graph descriptor for the graph."};
        }

        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_set_attribute(_graph_desc->get(),
                                                   HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                                   HIPDNN_TYPE_HANDLE,
                                                   1,
                                                   &handle),
            "Failed to set handle on the graph.");

        RETURN_ON_BACKEND_FAILURE(hipdnn_backend().backend_finalize(_graph_desc->get()),
                                  "Failed to finalize backend descriptor for the graph");

        return {error_code_t::OK, ""};
    }

    error_t create_execution_plans(hipdnnHandle_t handle,
                                   std::vector<HeurMode_t> const& modes = {HeurMode_t::FALLBACK})
    {
        if(!_graph_desc || !_graph_desc->valid())
        {
            return {error_code_t::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot create "
                    "execution plan."};
        }

        error_t status = initialize_heuristic_descriptor(modes);
        CHECK_HIPDNN_ERROR(status);

        status = initialize_engine_config();
        CHECK_HIPDNN_ERROR(status);

        _execution_plan_desc = std::make_unique<Scoped_hipdnn_backend_descriptor>(
            HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);

        if(!_execution_plan_desc->valid())
        {
            return {error_code_t::HIPDNN_BACKEND_ERROR,
                    "Failed to create backend execution descriptor."};
        }

        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_set_attribute(_execution_plan_desc->get(),
                                                   HIPDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                                   HIPDNN_TYPE_HANDLE,
                                                   1,
                                                   &handle),
            "Failed to set the handle on execution plan.");

        return {error_code_t::OK, ""};
    }

    error_t check_support()
    {
        if(!_execution_plan_desc || !_execution_plan_desc->valid())
        {
            return {error_code_t::HIPDNN_BACKEND_ERROR,
                    "Execution plan descriptor is not created or invalid."};
        }

        return {error_code_t::OK, ""};
    }

    error_t build_plans()
    {
        RETURN_ON_BACKEND_FAILURE(hipdnn_backend().backend_finalize(_engine_config_desc->get()),
                                  "Failed to finalize engine config descriptor");

        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_set_attribute(_execution_plan_desc->get(),
                                                   HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_engine_config_desc->get()),
            "Failed to set the engine config on execution plan.");

        RETURN_ON_BACKEND_FAILURE(hipdnn_backend().backend_finalize(_execution_plan_desc->get()),
                                  "Failed to finalize execution plan descriptor");

        return {error_code_t::OK, ""};
    }

    error_t get_workspace_size(int64_t& workspace_size) const
    {
        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_get_attribute(_execution_plan_desc->get(),
                                                   HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                                   HIPDNN_TYPE_INT64,
                                                   1,
                                                   nullptr,
                                                   &workspace_size),
            "Failed to get engine configurations from the execution plan descriptor.");

        return {error_code_t::OK, ""};
    }

    error_t execute(hipdnnHandle_t handle,
                    std::unordered_map<int64_t, void*>& variant_pack,
                    void* workspace) const
    {
        auto variant_pack_desc = std::make_unique<Scoped_hipdnn_backend_descriptor>(
            HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        if(!variant_pack_desc || !variant_pack_desc->valid())
        {
            return {error_code_t::HIPDNN_BACKEND_ERROR,
                    "Failed to create variant pack descriptor."};
        }

        //split variant_pack into vector of keys and vector of values
        std::vector<int64_t> variant_pack_keys;
        std::vector<void*> variant_pack_values;
        variant_pack_keys.reserve(variant_pack.size());
        variant_pack_values.reserve(variant_pack.size());
        for(const auto& [key, value] : variant_pack)
        {
            variant_pack_keys.push_back(key);
            variant_pack_values.push_back(value);
        }

        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_set_attribute(variant_pack_desc->get(),
                                                   HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                   HIPDNN_TYPE_VOID_PTR,
                                                   static_cast<int64_t>(variant_pack_values.size()),
                                                   variant_pack_values.data()),
            "failed to set the variant pack data pointers.");

        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_set_attribute(variant_pack_desc->get(),
                                                   HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                                   HIPDNN_TYPE_INT64,
                                                   static_cast<int64_t>(variant_pack_keys.size()),
                                                   variant_pack_keys.data()),
            "failed to set the variant pack unique ids.");

        RETURN_ON_BACKEND_FAILURE(
            hipdnn_backend().backend_set_attribute(variant_pack_desc->get(),
                                                   HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                                   HIPDNN_TYPE_VOID_PTR,
                                                   1,
                                                   &workspace),
            "failed to set the variant pack unique ids.");

        RETURN_ON_BACKEND_FAILURE(hipdnn_backend().backend_finalize(variant_pack_desc->get()),
                                  "Failed to finalize variant pack descriptor");

        hipdnn_backend().backend_execute(
            handle, _execution_plan_desc->get(), variant_pack_desc->get());

        return {error_code_t::OK, ""};
    }

    const std::string& get_name() const
    {
        return graph_attributes.get_name();
    }
    DataType_t get_compute_data_type() const
    {
        return graph_attributes.get_compute_data_type();
    }
    DataType_t get_intermediate_data_type() const
    {
        return graph_attributes.get_intermediate_data_type();
    }
    DataType_t get_io_data_type() const
    {
        return graph_attributes.get_io_data_type();
    }

    // Forwarding setters
    Graph& set_name(const std::string& name)
    {
        graph_attributes.set_name(name);
        return *this;
    }
    Graph& set_compute_data_type(DataType_t compute_type)
    {
        graph_attributes.set_compute_data_type(compute_type);
        return *this;
    }
    Graph& set_intermediate_data_type(DataType_t intermediate_type)
    {
        graph_attributes.set_intermediate_data_type(intermediate_type);
        return *this;
    }
    Graph& set_io_data_type(DataType_t io_type)
    {
        graph_attributes.set_io_data_type(io_type);
        return *this;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 5>
        batchnorm(const std::shared_ptr<Tensor_attributes>& x,
                  const std::shared_ptr<Tensor_attributes>& scale,
                  const std::shared_ptr<Tensor_attributes>& bias,
                  Batchnorm_attributes attributes)
    {
        auto y = output_tensor(attributes.name + "::Y");
        auto mean_out = output_tensor(attributes.name + "::MEAN");
        auto inv_variance_out = output_tensor(attributes.name + "::INV_VARIANCE");

        auto prev_running_mean = attributes.get_prev_running_mean();
        auto prev_running_variance = attributes.get_prev_running_variance();
        auto momentum = attributes.get_momentum();

        std::shared_ptr<Tensor_attributes> next_running_mean;
        std::shared_ptr<Tensor_attributes> next_running_variance;
        if(prev_running_mean && prev_running_variance && momentum)
        {
            next_running_mean = output_tensor(attributes.name + "::NEXT_RUNNING_MEAN");
            next_running_variance = output_tensor(attributes.name + "::NEXT_RUNNING_VARIANCE");
        }

        attributes.set_x(x);
        attributes.set_scale(scale);
        attributes.set_bias(bias);
        attributes.set_y(y);
        attributes.set_mean(mean_out);
        attributes.set_inv_variance(inv_variance_out);
        attributes.set_next_running_mean(next_running_mean);
        attributes.set_next_running_variance(next_running_variance);

        _sub_nodes.emplace_back(
            std::make_shared<BatchnormNode>(std::move(attributes), graph_attributes));

        return {y, mean_out, inv_variance_out, next_running_mean, next_running_variance};
    }

    std::array<std::shared_ptr<Tensor_attributes>, 3>
        batchnorm_backward(const std::shared_ptr<Tensor_attributes>& dy,
                           const std::shared_ptr<Tensor_attributes>& x,
                           const std::shared_ptr<Tensor_attributes>& scale,
                           Batchnorm_backward_attributes attributes)
    {
        auto dx = output_tensor(attributes.name + "::DX");
        attributes.set_dx(dx);

        auto dscale = output_tensor(attributes.name + "::DSCALE");
        attributes.set_dscale(dscale);

        auto dbias = output_tensor(attributes.name + "::DBIAS");
        attributes.set_dbias(dbias);

        attributes.set_x(x);
        attributes.set_dy(dy);
        attributes.set_scale(scale);

        _sub_nodes.emplace_back(
            std::make_shared<BatchnormBackwardNode>(std::move(attributes), graph_attributes));

        return {dx, dscale, dbias};
    }

    std::shared_ptr<Tensor_attributes>
        batchnorm_inference(const std::shared_ptr<Tensor_attributes>& x,
                            const std::shared_ptr<Tensor_attributes>& mean,
                            const std::shared_ptr<Tensor_attributes>& inv_variance,
                            const std::shared_ptr<Tensor_attributes>& scale,
                            const std::shared_ptr<Tensor_attributes>& bias,
                            Batchnorm_inference_attributes attributes)
    {
        auto y = attributes.outputs[Batchnorm_inference_attributes::output_names::y]
            = output_tensor(attributes.name + "::Y");
        attributes.inputs[Batchnorm_inference_attributes::input_names::x] = x;
        attributes.inputs[Batchnorm_inference_attributes::input_names::mean] = mean;
        attributes.inputs[Batchnorm_inference_attributes::input_names::inv_variance] = inv_variance;
        attributes.inputs[Batchnorm_inference_attributes::input_names::scale] = scale;
        attributes.inputs[Batchnorm_inference_attributes::input_names::bias] = bias;

        _sub_nodes.emplace_back(
            std::make_shared<BatchnormInferenceNode>(std::move(attributes), graph_attributes));

        return y;
    }

    std::shared_ptr<Tensor_attributes> pointwise(const std::shared_ptr<Tensor_attributes>& in_0,
                                                 Pointwise_attributes attributes)

    {
        if(attributes.name.empty())
        {
            attributes.name = "Pointwise_" + std::to_string(_sub_nodes.size());
        }
        if(in_0->get_name().empty())
        {
            in_0->set_name(attributes.name + "::IN_0");
        }
        auto out_0 = attributes.outputs[Pointwise_attributes::output_names::out_0]
            = output_tensor(attributes.name + "::OUT_0");
        attributes.inputs[Pointwise_attributes::input_names::in_0] = in_0;

        _sub_nodes.emplace_back(
            std::make_shared<PointwiseNode>(std::move(attributes), graph_attributes));

        return out_0;
    }

    std::shared_ptr<Tensor_attributes> pointwise(const std::shared_ptr<Tensor_attributes>& in_0,
                                                 const std::shared_ptr<Tensor_attributes>& in_1,
                                                 Pointwise_attributes attributes)

    {
        if(attributes.name.empty())
        {
            attributes.name = "Pointwise_" + std::to_string(_sub_nodes.size());
        }
        if(in_0->get_name().empty())
        {
            in_0->set_name(attributes.name + "::IN_0");
        }
        if(in_1->get_name().empty())
        {
            in_1->set_name(attributes.name + "::IN_1");
        }
        auto out_0 = attributes.outputs[Pointwise_attributes::output_names::out_0]
            = output_tensor(attributes.name + "::OUT_0");
        attributes.inputs[Pointwise_attributes::input_names::in_0] = in_0;
        attributes.inputs[Pointwise_attributes::input_names::in_1] = in_1;

        _sub_nodes.emplace_back(
            std::make_shared<PointwiseNode>(std::move(attributes), graph_attributes));

        return out_0;
    }

    std::shared_ptr<Tensor_attributes> pointwise(const std::shared_ptr<Tensor_attributes>& in_0,
                                                 const std::shared_ptr<Tensor_attributes>& in_1,
                                                 const std::shared_ptr<Tensor_attributes>& in_2,
                                                 Pointwise_attributes attributes)

    {
        if(attributes.name.empty())
        {
            attributes.name = "Pointwise_" + std::to_string(_sub_nodes.size());
        }
        if(in_0->get_name().empty())
        {
            in_0->set_name(attributes.name + "::IN_0");
        }
        if(in_1->get_name().empty())
        {
            in_1->set_name(attributes.name + "::IN_1");
        }
        if(in_2->get_name().empty())
        {
            in_2->set_name(attributes.name + "::IN_2");
        }
        auto out_0 = attributes.outputs[Pointwise_attributes::output_names::out_0]
            = output_tensor(attributes.name + "::OUT_0");
        attributes.inputs[Pointwise_attributes::input_names::in_0] = in_0;
        attributes.inputs[Pointwise_attributes::input_names::in_1] = in_1;
        attributes.inputs[Pointwise_attributes::input_names::in_2] = in_2;

        _sub_nodes.emplace_back(
            std::make_shared<PointwiseNode>(std::move(attributes), graph_attributes));

        return out_0;
    }

    static std::shared_ptr<Tensor_attributes>
        tensor_like(const std::shared_ptr<Tensor_attributes>& tensor, const std::string& name = "")
    {
        auto new_tensor = std::make_shared<Tensor_attributes>(*tensor);

        new_tensor->clear_uid();
        new_tensor->set_name(name);

        return new_tensor;
    }

    static std::shared_ptr<Tensor_attributes> tensor(const Tensor_attributes& tensor)
    {
        auto new_tensor = std::make_shared<Tensor_attributes>(tensor);

        return new_tensor;
    }
};
}
}
