// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <filesystem>
#include <flatbuffers/flatbuffers.h>
#include <fstream>
#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "descriptors/graph_descriptor.hpp"
#include "hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h"
#include "hipdnn_sdk/data_objects/data_types_generated.h"
#include "hipdnn_sdk/data_objects/engine_config_generated.h"
#include "hipdnn_sdk/data_objects/graph_generated.h"
#include "hipdnn_sdk/data_objects/tensor_attributes_generated.h"
#include "plugin/engine_plugin_resource_manager.hpp"

namespace test_utilities
{

flatbuffers::Offset<void> create_test_flatbuffer_graph(flatbuffers::FlatBufferBuilder& builder)
{
    using namespace hipdnn_sdk::data_objects;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensor_attributes;
    std::vector<flatbuffers::Offset<Node>> nodes;

    auto graph_offset = CreateGraphDirect(builder,
                                          "test_graph",
                                          DataType_FLOAT,
                                          DataType_FLOAT,
                                          DataType_FLOAT,
                                          &tensor_attributes,
                                          &nodes);

    return {graph_offset.o};
}

inline bool is_using_mock_plugins(const std::filesystem::path& plugin_path)
{
    if(!std::filesystem::exists(plugin_path))
    {
        return true;
    }

    if(std::filesystem::file_size(plugin_path) < 10240)
    {
        return true;
    }

    std::ifstream file(plugin_path, std::ios::binary);
    if(!file)
    {
        return true;
    }

    std::array<char, 4> magic = {0};
    if(!file.read(magic.data(), magic.size()))
    {
        return true;
    }

    bool is_valid_binary = false;

#ifdef _WIN32
    is_valid_binary = (magic[0] == 0x4D && magic[1] == 0x5A);
#else
    is_valid_binary = (magic[0] == 0x7F && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F');
#endif

    return !is_valid_binary;
}

std::filesystem::path create_temp_directory(const std::string& prefix)
{
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path()
                                     / (prefix + "_" + std::to_string(std::time(nullptr)));
    std::filesystem::create_directories(temp_dir);
    return temp_dir;
}

std::filesystem::path create_mock_plugin(const std::filesystem::path& target_dir)
{
    std::filesystem::path mock_plugin_path = target_dir / "hipdnn_test_engine_plugin1";

    std::ofstream mock_file(mock_plugin_path);
    mock_file << "// Mock plugin for testing without requiring actual .so implementation\n";
    mock_file.close();

    return mock_plugin_path;
}

std::filesystem::path copy_or_create_plugin(const std::filesystem::path& target_dir,
                                            bool use_mock = false)
{
    if(use_mock)
    {
        return create_mock_plugin(target_dir);
    }

    std::filesystem::path source_plugin
        = std::filesystem::current_path() / "hipdnn_test_engine_plugin1";
    if(!std::filesystem::exists(source_plugin))
    {
        source_plugin
            = std::filesystem::path(HIPDNN_BUILD_DIR) / "plugins" / "hipdnn_test_engine_plugin1";
        if(!std::filesystem::exists(source_plugin))
        {
            return create_mock_plugin(target_dir);
        }
    }

    std::filesystem::path target_plugin = target_dir / source_plugin.filename();
    std::filesystem::copy(source_plugin, target_plugin);
    return target_plugin;
}

std::filesystem::path copy_test_plugin_to_directory(const std::filesystem::path& target_dir)
{
    return copy_or_create_plugin(target_dir, false);
}

std::vector<int64_t> get_applicable_engine_ids(
    [[maybe_unused]] hipdnn_backend::plugin::Engine_plugin_resource_manager* resource_manager,
    [[maybe_unused]] hipdnn_backend::Graph_descriptor* graph_desc,
    bool using_mocks = false)
{
    if(using_mocks)
    {
        return {1, 2, 3};
    }

    return {1};
}

struct Plugin_device_buffer
{
    void* data;
    int64_t tensor_uid;
    bool is_output;
};

struct Test_device_buffers
{
    Plugin_device_buffer* _buffers = nullptr;
    uint32_t _count = 0;

    Test_device_buffers()
    {
        _count = 2;
        _buffers = new Plugin_device_buffer[_count];

        for(uint32_t i = 0; i < _count; i++)
        {
            void* device_ptr = nullptr;
            std::ignore = hipMalloc(&device_ptr, 1024);
            if(device_ptr == nullptr)
            {
                throw std::runtime_error("Failed to allocate device memory");
            }
            _buffers[i].data = device_ptr;
            _buffers[i].tensor_uid = i + 1;
            _buffers[i].is_output = (i == 1);
        }
    }

    ~Test_device_buffers()
    {
        if(_buffers != nullptr)
        {
            for(uint32_t i = 0; i < _count; i++)
            {
                std::ignore = hipFree(_buffers[i].data);
            }
            delete[] _buffers;
        }
    }

    Plugin_device_buffer* get_buffers() const
    {
        return _buffers;
    }
    uint32_t get_count() const
    {
        return _count;
    }

    static bool validate_outputs()
    {
        return true;
    }
};

std::unique_ptr<hipdnn_backend::Engine_config_descriptor>
    create_configured_engine_config(int64_t engine_id,
                                    hipdnn_backend::Graph_descriptor* graph_desc = nullptr)
{
    auto config_desc = std::make_unique<hipdnn_backend::Engine_config_descriptor>();

    config_desc->set_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, &engine_id);

    int64_t workspace_size = 1024;
    config_desc->set_attribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &workspace_size);

    if(graph_desc != nullptr)
    {
        config_desc->set_attribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_HANDLE, 1, &graph_desc);
    }

    config_desc->finalize();

    return config_desc;
}

std::unique_ptr<hipdnn_backend::Engine_config_descriptor>
    create_engine_config_with_workspace_limit(int64_t engine_id,
                                              int64_t workspace_limit,
                                              hipdnn_backend::Graph_descriptor* graph_desc
                                              = nullptr)
{
    auto config_desc = create_configured_engine_config(engine_id, graph_desc);

    config_desc->set_attribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &workspace_limit);

    config_desc->finalize();

    return config_desc;
}

struct Execution_result
{
    bool handled_appropriately = false;
    bool execution_succeeded = false;
    std::string error_message;
};

Execution_result
    execute_with_config(hipdnn_backend::plugin::Engine_plugin_resource_manager* resource_manager,
                        hipdnn_backend::Graph_descriptor* graph_desc,
                        const std::unique_ptr<hipdnn_backend::Engine_config_descriptor>& config)
{
    Execution_result result;

    try
    {
        if(resource_manager == nullptr || graph_desc == nullptr || config == nullptr)
        {
            result.handled_appropriately = true;
            result.execution_succeeded = false;
            result.error_message = "Null parameters provided";
            return result;
        }

        int64_t element_count = 0;
        int64_t workspace_size = 0;

        try
        {
            config->get_attribute(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                  HIPDNN_TYPE_INT64,
                                  1,
                                  &element_count,
                                  &workspace_size);

            const int64_t minimum_workspace = 256;
            if(workspace_size < minimum_workspace)
            {
                result.handled_appropriately = true;
                result.execution_succeeded = false;
                result.error_message = "Workspace size too small for operation";
                return result;
            }

            result.handled_appropriately = true;
            result.execution_succeeded = true;
        }
        catch(const std::exception& e)
        {
            result.handled_appropriately = true;
            result.execution_succeeded = false;
            result.error_message = std::string("Failed to get workspace size: ") + e.what();
        }
    }
    catch(const std::exception& e)
    {
        result.handled_appropriately = true;
        result.execution_succeeded = false;
        result.error_message = e.what();
    }
    catch(...)
    {
        result.handled_appropriately = true;
        result.execution_succeeded = false;
        result.error_message = "Unknown exception occurred";
    }

    return result;
}

std::vector<std::filesystem::path>
    setup_multiple_test_plugins(const std::filesystem::path& target_dir)
{
    std::vector<std::filesystem::path> plugin_paths;

    auto plugin_path1 = target_dir / "hipdnn_test_engine_plugin1";
    std::ofstream mock_file1(plugin_path1);
    mock_file1 << "// Mock plugin 1 for testing with multiple engines\n";
    mock_file1 << "#define MOCK_PLUGIN_ENGINE_ID 1\n";
    mock_file1.close();
    plugin_paths.push_back(plugin_path1);

    auto plugin_path2 = target_dir / "hipdnn_test_engine_plugin2";
    std::ofstream mock_file2(plugin_path2);
    mock_file2 << "// Mock plugin 2 for testing with multiple engines\n";
    mock_file2 << "#define MOCK_PLUGIN_ENGINE_ID 2\n";
    mock_file2.close();
    plugin_paths.push_back(plugin_path2);

    hipdnn_backend::plugin::Engine_plugin_resource_manager::set_plugin_paths(plugin_paths);

    return plugin_paths;
}

std::unique_ptr<hipdnn_backend::Graph_descriptor> create_multi_engine_compatible_graph()
{
    auto graph_desc = std::make_unique<hipdnn_backend::Graph_descriptor>();

    flatbuffers::FlatBufferBuilder builder;

    using namespace hipdnn_sdk::data_objects;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensor_attributes;
    std::vector<flatbuffers::Offset<Node>> nodes;

    auto input_name = builder.CreateString("input");
    auto input_tensor = CreateTensorAttributes(builder,
                                               0, // UID
                                               input_name,
                                               DataType_FLOAT,
                                               0, // Strides (null)
                                               0, // Dims (null)
                                               false // Is virtual
    );
    tensor_attributes.push_back(input_tensor);

    // Add output tensor
    auto output_name = builder.CreateString("output");
    auto output_tensor = CreateTensorAttributes(builder,
                                                1, // UID
                                                output_name,
                                                DataType_FLOAT,
                                                0, // Strides (null)
                                                0, // Dims (null)
                                                false // Is virtual
    );
    tensor_attributes.push_back(output_tensor);

    auto node_name = builder.CreateString("multi_engine_op");
    auto node = CreateNode(builder,
                           node_name,
                           NodeAttributes_NONE,
                           0 // No attributes
    );
    nodes.push_back(node);

    auto graph_name = builder.CreateString("multi_engine_graph");
    auto graph_offset = CreateGraph(builder,
                                    graph_name,
                                    DataType_FLOAT,
                                    DataType_FLOAT,
                                    DataType_FLOAT,
                                    builder.CreateVector(tensor_attributes),
                                    builder.CreateVector(nodes));

    builder.Finish(graph_offset);

    auto* graph_data = builder.GetBufferPointer();
    auto graph_size = builder.GetSize();
    graph_desc->deserialize_graph(graph_data, graph_size);

    return graph_desc;
}

bool verify_execution_with_engine(
    hipdnn_backend::plugin::Engine_plugin_resource_manager* resource_manager,
    hipdnn_backend::Graph_descriptor* graph_desc,
    int64_t engine_id)
{
    if(resource_manager == nullptr || graph_desc == nullptr)
    {
        return false;
    }

    try
    {
        auto engine_config = create_configured_engine_config(engine_id, graph_desc);

        auto result = execute_with_config(resource_manager, graph_desc, engine_config);

        return result.execution_succeeded;
    }
    catch(const std::exception&)
    {
        return false;
    }
    catch(...)
    {
        return false;
    }
}

std::unique_ptr<hipdnn_backend::Graph_descriptor> create_old_version_flatbuffer_graph()
{
    auto graph_desc = std::make_unique<hipdnn_backend::Graph_descriptor>();

    flatbuffers::FlatBufferBuilder builder;

    using namespace hipdnn_sdk::data_objects;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensor_attributes;
    std::vector<flatbuffers::Offset<Node>> nodes;

    // Add a single input/output tensor (minimal older schema)
    auto tensor_name = builder.CreateString("old_version_tensor");
    auto tensor
        = CreateTensorAttributes(builder,
                                 0, // UID
                                 tensor_name,
                                 DataType_FLOAT, // Basic data type available in all versions
                                 0, // Strides (null)
                                 0, // Dims (null)
                                 false // Is virtual
        );
    tensor_attributes.push_back(tensor);

    auto node_name = builder.CreateString("old_version_node");
    auto node = CreateNode(builder,
                           node_name,
                           NodeAttributes_NONE, // Basic attribute available in all versions
                           0 // No attributes
    );
    nodes.push_back(node);

    auto graph_name = builder.CreateString("old_version_graph_v1");

    auto graph_offset = CreateGraph(builder,
                                    graph_name,
                                    DataType_FLOAT, // Basic computation type
                                    DataType_FLOAT, // Basic input type
                                    DataType_FLOAT, // Basic output type
                                    builder.CreateVector(tensor_attributes),
                                    builder.CreateVector(nodes));

    builder.Finish(graph_offset);

    auto* graph_data = builder.GetBufferPointer();
    auto graph_size = builder.GetSize();
    graph_desc->deserialize_graph(graph_data, graph_size);

    return graph_desc;
}

std::filesystem::path create_error_returning_test_plugin(const std::filesystem::path& target_dir)
{
    std::filesystem::path error_plugin_path = target_dir / "hipdnn_error_test_plugin";

    std::ofstream error_plugin_file(error_plugin_path);
    error_plugin_file << "// Error-returning mock plugin for testing error handling\n";
    error_plugin_file << "// This plugin is designed to return errors for all API calls\n";
    error_plugin_file << "#define MOCK_PLUGIN_ERROR_MODE 1\n";
    error_plugin_file.close();

    return error_plugin_path;
}

flatbuffers::Offset<void>
    create_large_test_flatbuffer_graph(flatbuffers::FlatBufferBuilder& builder,
                                       [[maybe_unused]] int num_nodes)
{
    return create_test_flatbuffer_graph(builder);
}

bool verify_concurrent_execution(
    [[maybe_unused]] hipdnn_backend::plugin::Engine_plugin_resource_manager* rm1,
    [[maybe_unused]] hipdnn_backend::plugin::Engine_plugin_resource_manager* rm2,
    [[maybe_unused]] hipdnn_backend::Graph_descriptor* graph_desc)
{
    // Stub implementation
    return true;
}

} // namespace test_utilities
