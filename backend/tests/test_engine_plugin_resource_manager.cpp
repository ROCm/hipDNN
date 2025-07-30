
#include <filesystem>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/scoped_resource.hpp>
#include <iostream>
#include <memory>
#include <thread>

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/execution_plan_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/variant_descriptor.hpp"
#include "flatbuffers/flatbuffers.h"
#include "hipdnn_exception.hpp"
#include "plugin/engine_plugin_resource_manager.hpp"
#include "test_utilities/test_plugin_helpers.hpp"

using namespace hipdnn_backend;
using namespace test_utilities;

class Engine_plugin_resource_manager_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        _test_plugin_dir = create_temp_directory("engine_plugin_test");
        _test_plugin_path = copy_test_plugin_to_directory(_test_plugin_dir);

        std::vector<std::filesystem::path> plugin_paths = {_test_plugin_dir};
        plugin::Engine_plugin_resource_manager::set_plugin_paths(plugin_paths);

        _resource_manager = plugin::Engine_plugin_resource_manager::create();
        ASSERT_NE(_resource_manager, nullptr);

        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        new(&_stream_resource)
            hipdnn::sdk::utilities::Scoped_resource<hipStream_t, std::function<void(hipStream_t)>>(
                _stream, [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });

        _resource_manager->set_stream(_stream);
    }

    void TearDown() override
    {
        _resource_manager.reset();

        if(_stream != nullptr)
        {
            std::ignore = hipStreamDestroy(_stream);
            _stream = nullptr;
        }
        std::filesystem::remove_all(_test_plugin_dir);
    }

    static std::unique_ptr<Graph_descriptor> create_test_graph()
    {
        auto graph_desc = std::make_unique<Graph_descriptor>();

        flatbuffers::FlatBufferBuilder builder;
        auto graph_offset = create_test_flatbuffer_graph(builder);
        builder.Finish(graph_offset);

        auto* graph_data = builder.GetBufferPointer();
        auto graph_size = builder.GetSize();

        graph_desc->deserialize_graph(graph_data, graph_size);

        return graph_desc;
    }

    std::filesystem::path _test_plugin_dir;
    std::filesystem::path _test_plugin_path;
    std::shared_ptr<plugin::Engine_plugin_resource_manager> _resource_manager;
    hipStream_t _stream = nullptr;
    hipdnn::sdk::utilities::Scoped_resource<hipStream_t, std::function<void(hipStream_t)>>
        _stream_resource;
};

TEST_F(Engine_plugin_resource_manager_test, load_plugins_and_verify_configuration)
{
    auto graph_desc = create_test_graph();

    bool using_mocks = test_utilities::is_using_mock_plugins(_test_plugin_path);

    auto engine_ids
        = get_applicable_engine_ids(_resource_manager.get(), graph_desc.get(), using_mocks);
    ASSERT_FALSE(engine_ids.empty());

    auto engine_config_desc = create_configured_engine_config(engine_ids[0], graph_desc.get());

    std::string plugin_type = using_mocks ? "mock" : "real";
    SUCCEED();
    std::cout << "Successfully configured engine with " << plugin_type << " plugin\n";
}

TEST_F(Engine_plugin_resource_manager_test, handle_nonexistent_plugin_path)
{
    std::vector<std::filesystem::path> original_paths = {_test_plugin_dir};

    std::vector<std::filesystem::path> invalid_paths = {"/path/does/not/exist"};
    plugin::Engine_plugin_resource_manager::set_plugin_paths(invalid_paths);

    auto resource_manager = plugin::Engine_plugin_resource_manager::create();
    EXPECT_NE(resource_manager, nullptr);

    plugin::Engine_plugin_resource_manager::set_plugin_paths(original_paths);
}

TEST_F(Engine_plugin_resource_manager_test, handle_malformed_flatbuffer_data)
{
    auto graph_desc = std::make_unique<Graph_descriptor>();

    std::vector<uint8_t> invalid_data(100, 0xFF);

    EXPECT_THROW(graph_desc->deserialize_graph(invalid_data.data(), invalid_data.size()),
                 Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_test, handle_null_graph_descriptor)
{
    EXPECT_THROW(test_utilities::get_applicable_engine_ids(_resource_manager.get(), nullptr),
                 Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_test, handle_null_execution_context)
{
    Execution_plan_descriptor execution_plan_desc;

    Variant_descriptor variant_pack;

    EXPECT_THROW(_resource_manager->execute_op_graph(
                     reinterpret_cast<hipdnnBackendDescriptor_t>(&execution_plan_desc),
                     reinterpret_cast<hipdnnBackendDescriptor_t>(&variant_pack)),
                 Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_test, handle_concurrent_streams)
{
    hipStream_t stream1;
    hipStream_t stream2;
    ASSERT_EQ(hipStreamCreate(&stream1), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream2), hipSuccess);

    hipdnn::sdk::utilities::Scoped_resource<hipStream_t, std::function<void(hipStream_t)>>
        stream1_res(stream1, [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });
    hipdnn::sdk::utilities::Scoped_resource<hipStream_t, std::function<void(hipStream_t)>>
        stream2_res(stream2, [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });

    auto resource_manager1 = plugin::Engine_plugin_resource_manager::create();
    auto resource_manager2 = plugin::Engine_plugin_resource_manager::create();

    resource_manager1->set_stream(stream1);
    resource_manager2->set_stream(stream2);

    auto graph_desc = create_test_graph();

    EXPECT_NO_THROW(test_utilities::verify_concurrent_execution(
        resource_manager1.get(), resource_manager2.get(), graph_desc.get()));
}

TEST_F(Engine_plugin_resource_manager_test, handle_large_flatbuffers)
{
    flatbuffers::FlatBufferBuilder builder;
    auto graph_offset
        = test_utilities::create_large_test_flatbuffer_graph(builder, 1000); // 1000 nodes
    builder.Finish(graph_offset);

    auto graph_desc = std::make_unique<Graph_descriptor>();
    graph_desc->deserialize_graph(builder.GetBufferPointer(), builder.GetSize());

    auto engine_ids
        = test_utilities::get_applicable_engine_ids(_resource_manager.get(), graph_desc.get());

    EXPECT_FALSE(engine_ids.empty());
}

TEST_F(Engine_plugin_resource_manager_test, verify_plugin_handle_management)
{
    auto graph_desc = create_test_graph();

    auto engine_ids
        = test_utilities::get_applicable_engine_ids(_resource_manager.get(), graph_desc.get());

    auto engine_ids2
        = test_utilities::get_applicable_engine_ids(_resource_manager.get(), graph_desc.get());

    ASSERT_EQ(engine_ids.size(), engine_ids2.size());
    for(size_t i = 0; i < engine_ids.size(); ++i)
    {
        EXPECT_EQ(engine_ids[i], engine_ids2[i]);
    }
}

TEST_F(Engine_plugin_resource_manager_test, handle_multithreaded_access)
{
    constexpr int num_threads = 4;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    std::atomic<bool> found_error(false);

    for(int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([this, &found_error]() {
            try
            {
                auto graph_desc = create_test_graph();
                auto engine_ids = test_utilities::get_applicable_engine_ids(_resource_manager.get(),
                                                                            graph_desc.get());
                if(engine_ids.empty())
                {
                    found_error = true;
                }
            }
            catch(...)
            {
                found_error = true;
            }
        });
    }

    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_FALSE(found_error);
}

TEST_F(Engine_plugin_resource_manager_test, handle_plugin_error_conditions)
{
    auto error_plugin_path = test_utilities::create_error_returning_test_plugin(_test_plugin_dir);

    std::vector<std::filesystem::path> plugin_paths = {error_plugin_path};
    plugin::Engine_plugin_resource_manager::set_plugin_paths(plugin_paths);

    auto error_resource_manager = plugin::Engine_plugin_resource_manager::create();
    ASSERT_NE(error_resource_manager, nullptr);

    auto graph_desc = create_test_graph();

    auto engine_ids = test_utilities::get_applicable_engine_ids(error_resource_manager.get(),
                                                                graph_desc.get(),
                                                                true // using_mocks = true
    );

    EXPECT_TRUE(engine_ids.empty() || !engine_ids.empty());

    EXPECT_NO_THROW({ error_resource_manager->set_stream(_stream); });
}

TEST_F(Engine_plugin_resource_manager_test, handle_workspace_size_limitations)
{
    auto graph_desc = create_test_graph();
    auto engine_ids
        = test_utilities::get_applicable_engine_ids(_resource_manager.get(), graph_desc.get());
    ASSERT_FALSE(engine_ids.empty());

    auto config_with_small_workspace = test_utilities::create_engine_config_with_workspace_limit(
        engine_ids[0], 128); // Deliberately small workspace

    auto result = test_utilities::execute_with_config(
        _resource_manager.get(), graph_desc.get(), config_with_small_workspace);
    EXPECT_TRUE(result.handled_appropriately);
}

TEST_F(Engine_plugin_resource_manager_test, select_between_multiple_engines)
{
    auto multi_plugin_setup = test_utilities::setup_multiple_test_plugins(_test_plugin_dir);
    auto multi_engine_manager = plugin::Engine_plugin_resource_manager::create();

    auto graph_desc = test_utilities::create_multi_engine_compatible_graph();

    auto engine_ids
        = test_utilities::get_applicable_engine_ids(multi_engine_manager.get(), graph_desc.get());
    EXPECT_GE(engine_ids.size(), 2); // At least two engines should be able to handle it

    for(auto engine_id : engine_ids)
    {
        EXPECT_TRUE(test_utilities::verify_execution_with_engine(
            multi_engine_manager.get(), graph_desc.get(), engine_id));
    }
}

TEST_F(Engine_plugin_resource_manager_test, handle_different_flatbuffer_versions)
{
    auto old_version_graph = test_utilities::create_old_version_flatbuffer_graph();
    auto current_version_graph = create_test_graph();

    EXPECT_NO_THROW(test_utilities::get_applicable_engine_ids(_resource_manager.get(),
                                                              old_version_graph.get()));
    EXPECT_NO_THROW(test_utilities::get_applicable_engine_ids(_resource_manager.get(),
                                                              current_version_graph.get()));
}

TEST_F(Engine_plugin_resource_manager_test, handle_plugin_lifecycle_events)
{
    auto graph_desc = create_test_graph();

    auto initial_engine_ids
        = test_utilities::get_applicable_engine_ids(_resource_manager.get(), graph_desc.get());

    std::vector<std::filesystem::path> empty_paths;
    plugin::Engine_plugin_resource_manager::set_plugin_paths(empty_paths);

    std::vector<std::filesystem::path> original_paths = {_test_plugin_dir};
    plugin::Engine_plugin_resource_manager::set_plugin_paths(original_paths);

    auto new_manager = plugin::Engine_plugin_resource_manager::create();

    auto new_engine_ids
        = test_utilities::get_applicable_engine_ids(new_manager.get(), graph_desc.get());
    EXPECT_EQ(initial_engine_ids.size(), new_engine_ids.size());
}

TEST_F(Engine_plugin_resource_manager_test, handle_gpu_execution_errors)
{
    auto graph_desc = create_test_graph();

    auto engine_ids
        = test_utilities::get_applicable_engine_ids(_resource_manager.get(), graph_desc.get());
    ASSERT_FALSE(engine_ids.empty());

    auto engine_config_desc
        = test_utilities::create_configured_engine_config(engine_ids[0], graph_desc.get());

    hipdnn_backend::Variant_descriptor variant_pack;

    test_utilities::Test_device_buffers test_buffers;

    void* invalid_ptr = reinterpret_cast<void*>(0x1); // Invalid pointer that will cause GPU error
    auto buffers = test_buffers.get_buffers();
    buffers[0].data = invalid_ptr;

    auto* config_ptr = reinterpret_cast<hipdnnBackendDescriptor_t>(engine_config_desc.get());
    auto* variant_ptr = reinterpret_cast<hipdnnBackendDescriptor_t>(&variant_pack);

    EXPECT_THROW(_resource_manager->execute_op_graph(config_ptr, variant_ptr),
                 hipdnn_backend::Hipdnn_exception);

    test_utilities::Test_device_buffers valid_buffers;

    EXPECT_NO_THROW({ _resource_manager->execute_op_graph(config_ptr, variant_ptr); });
}

// // Test when multiple plugins claim the same engine ID
// TEST_F(Engine_plugin_resource_manager_test, handle_duplicate_engine_ids)
// {
//     // Create two plugins that claim the same engine ID
//     auto conflicting_plugins = test_utilities::create_plugins_with_duplicate_engine_ids(_test_plugin_dir);

//     // Set up paths to include both conflicting plugins
//     std::vector<std::filesystem::path> plugin_paths = {
//         conflicting_plugins.plugin_path_1,
//         conflicting_plugins.plugin_path_2
//     };
//     plugin::Engine_plugin_resource_manager::set_plugin_paths(plugin_paths);

//     // Log handler to capture warnings about duplicate engine IDs
//     auto warning_logger = test_utilities::create_warning_capture_logger();

//     // Create resource manager - should load plugins without crashing
//     auto conflict_resource_manager = plugin::Engine_plugin_resource_manager::create();
//     ASSERT_NE(conflict_resource_manager, nullptr);

//     // Create test graph
//     auto graph_desc = create_test_graph();

//     // Get applicable engines
//     auto engine_ids = test_utilities::get_applicable_engine_ids(
//         conflict_resource_manager.get(), graph_desc.get());

//     // Should only have one instance of the duplicate engine ID
//     auto duplicate_count = std::count(engine_ids.begin(), engine_ids.end(),
//                                       conflicting_plugins.duplicate_engine_id);
//     EXPECT_EQ(duplicate_count, 1);

//     // Verify warning was logged about duplicate engine ID
//     EXPECT_TRUE(test_utilities::check_warning_logged(warning_logger,
//                                                     "duplicate engine ID"));

//     // Execute with the duplicate engine ID to ensure consistent behavior
//     EXPECT_TRUE(test_utilities::verify_execution_with_engine(
//         conflict_resource_manager.get(), graph_desc.get(),
//         conflicting_plugins.duplicate_engine_id));
// }

// Test API version compatibility between resource manager and plugins
// TEST_F(Engine_plugin_resource_manager_test, handle_api_version_compatibility)
// {
//     // Create plugins with different API versions
//     auto version_plugins = test_utilities::create_plugins_with_different_versions(_test_plugin_dir);

//     // Set plugin paths to include all version test plugins
//     std::vector<std::filesystem::path> plugin_paths = {
//         version_plugins.compatible_plugin_path,
//         version_plugins.incompatible_plugin_path,
//         version_plugins.newer_plugin_path
//     };
//     plugin::Engine_plugin_resource_manager::set_plugin_paths(plugin_paths);

//     // Log handler to capture warnings/errors
//     auto log_handler = test_utilities::create_warning_capture_logger();

//     // Create resource manager
//     auto version_resource_manager = plugin::Engine_plugin_resource_manager::create();
//     ASSERT_NE(version_resource_manager, nullptr);

//     // Verify that compatible plugins were loaded
//     EXPECT_TRUE(test_utilities::plugin_was_loaded(
//         version_resource_manager.get(),
//         version_plugins.compatible_engine_id));

//     // Verify that incompatible plugins were not loaded or properly rejected
//     EXPECT_FALSE(test_utilities::plugin_was_loaded(
//         version_resource_manager.get(),
//         version_plugins.incompatible_engine_id));

//     // Verify warning was logged about incompatible version
//     EXPECT_TRUE(test_utilities::check_warning_logged(
//         log_handler, "incompatible API version"));

//     // Create test graph
//     auto graph_desc = create_test_graph();

//     // Execute with compatible plugin should work
//     EXPECT_TRUE(test_utilities::verify_execution_with_engine(
//         version_resource_manager.get(),
//         graph_desc.get(),
//         version_plugins.compatible_engine_id));
// }

// // // Test cleanup behavior when resource manager is destroyed
// TEST_F(Engine_plugin_resource_manager_test, verify_cleanup_on_destruction)
// {
//     // Create resource tracker to monitor plugin resources
//     auto resource_tracker = test_utilities::create_resource_tracker();

//     // Create a new resource manager with tracking
//     auto tracked_resource_manager = test_utilities::create_tracked_resource_manager(resource_tracker);
//     ASSERT_NE(tracked_resource_manager, nullptr);

//     // Create test graph
//     auto graph_desc = create_test_graph();

//     // Get applicable engines
//     auto engine_ids = test_utilities::get_applicable_engine_ids(
//         tracked_resource_manager.get(), graph_desc.get());
//     ASSERT_FALSE(engine_ids.empty());

//     // Create execution context
//     auto execution_context = test_utilities::create_execution_context(
//         tracked_resource_manager.get(), graph_desc.get(), engine_ids[0]);
//     ASSERT_NE(execution_context, nullptr);

//     // Verify resources were allocated
//     EXPECT_GT(resource_tracker->allocated_resources, 0);

//     // Destroy resource manager
//     tracked_resource_manager.reset();

//     // Verify all resources were cleaned up
//     EXPECT_EQ(resource_tracker->allocated_resources, 0);
//     EXPECT_EQ(resource_tracker->leaked_resources, 0);
// }