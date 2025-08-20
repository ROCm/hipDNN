// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <atomic>
#include <filesystem>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

#include "descriptors/flatbuffer_test_utils.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/mocks/mock_descriptor.hpp"
#include "descriptors/test_macros.hpp"
#include "plugin/engine_plugin_resource_manager.hpp"
#include "plugins/mocks/mock_engine_plugin.hpp"
#include "plugins/mocks/mock_engine_plugin_manager.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace ::testing;

TEST(Engine_plugin_resource_manager, plugin_loading)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};

    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));

    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);
    }
}

TEST(Engine_plugin_resource_manager, set_stream)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};

    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));

    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));

    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(*mock_plugin,
                set_stream(hipdnnEnginePluginHandle_t(0xdeadbeef), hipStream_t(0x12345678)));

    EXPECT_CALL(*mock_plugin, destroy_handle(hipdnnEnginePluginHandle_t(0xdeadbeef)));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        resource_manager.set_stream(hipStream_t(0x12345678));
    }
}

TEST(Engine_plugin_resource_manager, static_plugin_path_management_set_and_get_single_path)
{
    std::vector<std::filesystem::path> plugin_paths = {"/test/plugin/path"};

    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();

    std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST(Engine_plugin_resource_manager, static_plugin_path_management_set_and_get_multiple_paths)
{
    std::vector<std::filesystem::path> plugin_paths
        = {"/test/plugin/path1", "/test/plugin/path2", "/test/plugin/path3"};

    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();

    std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST(Engine_plugin_resource_manager, static_plugin_path_management_additive_loading_mode)
{
    std::vector<std::filesystem::path> initial_paths = {"/test/path1"};
    Engine_plugin_resource_manager::set_plugin_paths(initial_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    std::vector<std::filesystem::path> additional_paths = {"/test/path2", "/test/path3"};
    Engine_plugin_resource_manager::set_plugin_paths(additional_paths,
                                                     HIPDNN_PLUGIN_LOADING_ADDITIVE);

    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();

    std::set<std::filesystem::path> expected_paths = {"/test/path1", "/test/path2", "/test/path3"};
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST(Engine_plugin_resource_manager,
     static_plugin_path_management_absolute_loading_mode_replaces_existing)
{
    std::vector<std::filesystem::path> initial_paths = {"/test/path1", "/test/path2"};
    Engine_plugin_resource_manager::set_plugin_paths(initial_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    std::vector<std::filesystem::path> new_paths = {"/test/path3", "/test/path4"};
    Engine_plugin_resource_manager::set_plugin_paths(new_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();

    std::set<std::filesystem::path> expected_paths(new_paths.begin(), new_paths.end());
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST(Engine_plugin_resource_manager, static_plugin_path_management_empty_paths_clearing)
{
    std::vector<std::filesystem::path> plugin_paths = {"/test/path1", "/test/path2"};
    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    std::vector<std::filesystem::path> empty_paths;
    Engine_plugin_resource_manager::set_plugin_paths(empty_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();
    EXPECT_TRUE(retrieved_paths.empty());
}

TEST(Engine_plugin_resource_manager, move_constructor)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mock_plugin, set_stream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    Engine_plugin_resource_manager rm1(plugin_manager);

    Engine_plugin_resource_manager rm2 = std::move(rm1);

    EXPECT_NO_THROW(rm2.set_stream(nullptr));
}

TEST(Engine_plugin_resource_manager, move_assignment)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin1 = std::make_shared<Mock_engine_plugin>();
    std::shared_ptr<Mock_engine_plugin> mock_plugin2 = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins1{mock_plugin1};
    std::vector<std::shared_ptr<Engine_plugin>> plugins2{mock_plugin2};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager1
        = std::make_shared<Mock_engine_plugin_manager>();
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager2
        = std::make_shared<Mock_engine_plugin_manager>();

    EXPECT_CALL(*plugin_manager1, get_plugins()).WillOnce(::testing::ReturnRef(plugins1));
    EXPECT_CALL(*mock_plugin1, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin1, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*mock_plugin1, set_stream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
    EXPECT_CALL(*mock_plugin1, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    EXPECT_CALL(*plugin_manager2, get_plugins()).WillOnce(::testing::ReturnRef(plugins2));
    EXPECT_CALL(*mock_plugin2, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xcafebabe)));
    EXPECT_CALL(*mock_plugin2, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{200}));
    EXPECT_CALL(*mock_plugin2, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xcafebabe))))
        .Times(testing::AtMost(1));

    Engine_plugin_resource_manager rm1(plugin_manager1);
    Engine_plugin_resource_manager rm2(plugin_manager2);

    rm2 = std::move(rm1);

    EXPECT_NO_THROW(rm2.set_stream(nullptr));
}

TEST(Engine_plugin_resource_manager, self_move_assignment)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    Mock_graph_descriptor mock_graph_desc;
    hipdnnPluginConstData_t fake_serialized_data
        = {reinterpret_cast<const void*>("fake_graph_data"), 15};

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101}));

    EXPECT_CALL(*mock_plugin, set_stream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr)).Times(2);

    EXPECT_CALL(mock_graph_desc, get_serialized_graph())
        .WillOnce(::testing::Return(fake_serialized_data));
    EXPECT_CALL(*mock_plugin,
                get_applicable_engine_ids(hipdnnEnginePluginHandle_t(0xdeadbeef), testing::_))
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101}));

    std::set<std::filesystem::path> expected_plugin_files = {"/path/to/plugin.so"};
    EXPECT_CALL(*plugin_manager, get_loaded_plugin_files())
        .WillOnce(::testing::ReturnRef(expected_plugin_files));

    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    Engine_plugin_resource_manager rm(plugin_manager);

    EXPECT_NO_THROW(rm.set_stream(nullptr));

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
    rm = std::move(rm);
#pragma clang diagnostic pop

    EXPECT_NO_THROW(rm.set_stream(nullptr));

    auto engine_ids = rm.get_applicable_engine_ids(&mock_graph_desc);
    EXPECT_EQ(engine_ids.size(), 2);
    EXPECT_EQ(engine_ids[0], 100);
    EXPECT_EQ(engine_ids[1], 101);

    size_t num_plugins = 0;
    size_t max_string_len = 0;
    EXPECT_NO_THROW(rm.get_loaded_plugin_files(&num_plugins, nullptr, &max_string_len));
    EXPECT_EQ(num_plugins, 1);
    EXPECT_GT(max_string_len, 0);
}

TEST(Engine_plugin_resource_manager, rapid_creation_destruction)
{
    const int num_iterations = 100;

    for(int i = 0; i < num_iterations; ++i)
    {
        auto plugin_manager = std::make_shared<Mock_engine_plugin_manager>();
        auto mock_plugin = std::make_shared<Mock_engine_plugin>();
        std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};

        EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
        EXPECT_CALL(*mock_plugin, create_handle())
            .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
        EXPECT_CALL(*mock_plugin, get_all_engine_ids())
            .WillOnce(::testing::Return(std::vector<int64_t>{100}));
        EXPECT_CALL(*mock_plugin, set_stream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
        EXPECT_CALL(*mock_plugin,
                    destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

        {
            Engine_plugin_resource_manager rm(plugin_manager);

            EXPECT_NO_THROW(rm.set_stream(nullptr));
        }
    }
}

TEST(Engine_plugin_resource_manager, concurrent_creation_and_public_methods)
{
    const size_t num_threads = 4;
    const size_t managers_per_thread = 10;
    std::vector<std::thread> threads;
    std::vector<std::vector<Engine_plugin_resource_manager*>> all_managers(num_threads);
    std::vector<std::vector<std::shared_ptr<Mock_engine_plugin_manager>>> all_plugin_managers(
        num_threads);
    std::vector<std::vector<std::shared_ptr<Mock_engine_plugin>>> all_mock_plugins(num_threads);
    std::vector<std::vector<std::vector<std::shared_ptr<Engine_plugin>>>> all_plugins(num_threads);
    std::atomic<size_t> successful_creations{0};

    threads.reserve(num_threads);

    for(size_t t = 0; t < num_threads; ++t)
    {
        all_managers[t].reserve(managers_per_thread);
        all_plugin_managers[t].reserve(managers_per_thread);
        all_mock_plugins[t].reserve(managers_per_thread);
        all_plugins[t].reserve(managers_per_thread);
    }

    for(size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([t,
                              &all_managers,
                              &all_plugin_managers,
                              &all_mock_plugins,
                              &all_plugins,
                              &successful_creations]() {
            for(size_t i = 0; i < managers_per_thread; ++i)
            {
                auto plugin_manager = std::make_shared<Mock_engine_plugin_manager>();
                auto mock_plugin = std::make_shared<Mock_engine_plugin>();
                std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};

                all_plugin_managers[t].push_back(plugin_manager);
                all_mock_plugins[t].push_back(mock_plugin);
                all_plugins[t].push_back(plugins);

                EXPECT_CALL(*plugin_manager, get_plugins())
                    .WillOnce(::testing::ReturnRef(all_plugins[t][i]));
                EXPECT_CALL(*mock_plugin, create_handle())
                    .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
                EXPECT_CALL(*mock_plugin, get_all_engine_ids())
                    .WillOnce(::testing::Return(
                        std::vector<int64_t>{static_cast<int64_t>(100 + (t * 1000) + i)}));
                EXPECT_CALL(*mock_plugin,
                            set_stream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
                EXPECT_CALL(*mock_plugin,
                            destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

                all_managers[t].push_back(new Engine_plugin_resource_manager(plugin_manager));
                successful_creations++;

                EXPECT_NO_THROW(all_managers[t].back()->set_stream(nullptr));
            }
        });
    }

    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(successful_creations.load(), num_threads * managers_per_thread);

    for(const auto& thread_managers : all_managers)
    {
        for(const auto& rm : thread_managers)
        {
            EXPECT_NE(rm, nullptr);
            delete rm;
        }
    }
}

TEST(Engine_plugin_resource_manager, get_applicable_engine_ids_null_graph_descriptor)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        EXPECT_HIPDNN_EXCEPTION(resource_manager.get_applicable_engine_ids(nullptr),
                                HIPDNN_STATUS_BAD_PARAM);
    }
}

TEST(Engine_plugin_resource_manager, set_null_stream)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mock_plugin, set_stream(hipdnnEnginePluginHandle_t(0xdeadbeef), nullptr));
    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        EXPECT_NO_THROW(resource_manager.set_stream(nullptr));
    }
}

TEST(Engine_plugin_resource_manager, get_applicable_engine_ids_with_loaded_plugin)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    Mock_graph_descriptor mock_graph_desc;
    hipdnnPluginConstData_t fake_serialized_data = {
        reinterpret_cast<const void*>("fake_graph_data"),
        15 // length of "fake_graph_data"
    };

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(mock_graph_desc, get_serialized_graph())
        .WillOnce(::testing::Return(fake_serialized_data));

    EXPECT_CALL(
        *mock_plugin,
        get_applicable_engine_ids(
            hipdnnEnginePluginHandle_t(0xdeadbeef),
            testing::Pointee(testing::AllOf(
                testing::Field(&hipdnnPluginConstData_t::ptr, fake_serialized_data.ptr),
                testing::Field(&hipdnnPluginConstData_t::size, fake_serialized_data.size)))))
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        auto engine_ids = resource_manager.get_applicable_engine_ids(&mock_graph_desc);

        EXPECT_EQ(engine_ids.size(), 3);
        EXPECT_EQ(engine_ids[0], 100);
        EXPECT_EQ(engine_ids[1], 101);
        EXPECT_EQ(engine_ids[2], 102);
    }
}

TEST(Engine_plugin_resource_manager, get_workspace_size)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    Mock_graph_descriptor mock_graph_desc;
    hipdnnPluginConstData_t fake_engine_config = {
        reinterpret_cast<const void*>("fake_config"),
        11 // length of "fake_config"
    };
    hipdnnPluginConstData_t fake_serialized_data = {
        reinterpret_cast<const void*>("fake_graph_data"),
        15 // length of "fake_graph_data"
    };

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(mock_graph_desc, get_serialized_graph())
        .WillOnce(::testing::Return(fake_serialized_data));

    EXPECT_CALL(
        *mock_plugin,
        get_workspace_size(
            hipdnnEnginePluginHandle_t(0xdeadbeef),
            testing::Pointee(testing::AllOf(
                testing::Field(&hipdnnPluginConstData_t::ptr, fake_engine_config.ptr),
                testing::Field(&hipdnnPluginConstData_t::size, fake_engine_config.size))),
            testing::Pointee(testing::AllOf(
                testing::Field(&hipdnnPluginConstData_t::ptr, fake_serialized_data.ptr),
                testing::Field(&hipdnnPluginConstData_t::size, fake_serialized_data.size)))))
        .WillOnce(::testing::Return(size_t(8192)));

    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        auto workspace_size
            = resource_manager.get_workspace_size(100, &fake_engine_config, &mock_graph_desc);

        EXPECT_EQ(workspace_size, 8192);
    }
}

TEST(Engine_plugin_resource_manager, get_engine_details)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    Mock_graph_descriptor mock_graph_desc;
    hipdnnPluginConstData_t fake_serialized_data = {
        reinterpret_cast<const void*>("fake_graph_data"),
        15 // length of "fake_graph_data"
    };

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(mock_graph_desc, get_serialized_graph())
        .WillOnce(::testing::Return(fake_serialized_data));

    EXPECT_CALL(*mock_plugin,
                get_engine_details(
                    hipdnnEnginePluginHandle_t(0xdeadbeef),
                    int64_t(100), // engine_id
                    testing::Pointee(testing::AllOf(
                        testing::Field(&hipdnnPluginConstData_t::ptr, fake_serialized_data.ptr),
                        testing::Field(&hipdnnPluginConstData_t::size,
                                       fake_serialized_data.size))), // op_graph
                    testing::_ // output engine_details
                    ))
        .WillOnce(testing::Invoke([](hipdnnEnginePluginHandle_t,
                                     int64_t engine_id,
                                     const hipdnnPluginConstData_t*,
                                     hipdnnPluginConstData_t* output) {
            // Create valid flatbuffer engine details
            static auto builder = flatbuffer_test_utils::create_valid_engine_details(engine_id);
            output->ptr = builder.GetBufferPointer();
            output->size = builder.GetSize();
        }));

    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        // Test get_engine_details functionality with valid flatbuffer data
        auto engine_details = Engine_plugin_resource_manager::get_engine_details(
            std::make_shared<Engine_plugin_resource_manager>(std::move(resource_manager)),
            100,
            &mock_graph_desc);

        // Verify that we got valid engine details
        EXPECT_NE(engine_details, nullptr);
        EXPECT_NE(engine_details->get(), nullptr);
        EXPECT_EQ(engine_details->get()->engine_id(), 100);
    }
}

TEST(Engine_plugin_resource_manager, create_execution_context)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    Mock_graph_descriptor mock_graph_desc;
    hipdnnPluginConstData_t fake_engine_config = {
        reinterpret_cast<const void*>("fake_config"),
        11 // length of "fake_config"
    };
    hipdnnPluginConstData_t fake_serialized_data = {
        reinterpret_cast<const void*>("fake_graph_data"),
        15 // length of "fake_graph_data"
    };

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));

    EXPECT_CALL(mock_graph_desc, get_serialized_graph())
        .WillOnce(::testing::Return(fake_serialized_data));

    EXPECT_CALL(
        *mock_plugin,
        create_execution_context(
            hipdnnEnginePluginHandle_t(0xdeadbeef),
            testing::Pointee(testing::AllOf(
                testing::Field(&hipdnnPluginConstData_t::ptr, fake_engine_config.ptr),
                testing::Field(&hipdnnPluginConstData_t::size, fake_engine_config.size))),
            testing::Pointee(testing::AllOf(
                testing::Field(&hipdnnPluginConstData_t::ptr, fake_serialized_data.ptr),
                testing::Field(&hipdnnPluginConstData_t::size, fake_serialized_data.size)))))
        .WillOnce(::testing::Return(hipdnnEnginePluginExecutionContext_t(0xcafebabe)));

    EXPECT_CALL(*mock_plugin,
                destroy_execution_context(hipdnnEnginePluginHandle_t(0xdeadbeef),
                                          hipdnnEnginePluginExecutionContext_t(0xcafebabe)));

    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        auto execution_context = Engine_plugin_resource_manager::create_execution_context(
            std::make_shared<Engine_plugin_resource_manager>(std::move(resource_manager)),
            100,
            &fake_engine_config,
            &mock_graph_desc);

        EXPECT_NE(execution_context, nullptr);
        EXPECT_NE(execution_context->get(), nullptr);
    }
}

TEST(Engine_plugin_resource_manager, execute_op_graph)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        EXPECT_HIPDNN_EXCEPTION(resource_manager.execute_op_graph(nullptr, nullptr),
                                HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(Engine_plugin_resource_manager, get_loaded_plugin_files)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    std::set<std::filesystem::path> expected_plugin_files
        = {"/path/to/plugin1.so", "/path/to/plugin2.so"};

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*plugin_manager, get_loaded_plugin_files())
        .Times(2)
        .WillRepeatedly(::testing::ReturnRef(expected_plugin_files));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        size_t num_plugins = 0;
        size_t max_string_len = 0;

        EXPECT_NO_THROW(
            resource_manager.get_loaded_plugin_files(&num_plugins, nullptr, &max_string_len));

        EXPECT_EQ(num_plugins, 2);
        EXPECT_GT(max_string_len, 0);

        std::vector<std::string> plugin_strings(num_plugins);
        std::vector<char*> plugin_paths(num_plugins);
        for(size_t i = 0; i < num_plugins; ++i)
        {
            plugin_strings[i].resize(max_string_len);
            plugin_paths[i] = plugin_strings[i].data();
        }

        EXPECT_NO_THROW(resource_manager.get_loaded_plugin_files(
            &num_plugins, plugin_paths.data(), &max_string_len));

        // Note: std::set ordering may differ, so we check that both paths are present
        std::set<std::string> returned_paths
            = {std::string(plugin_paths[0]), std::string(plugin_paths[1])};
        std::set<std::string> expected_paths = {"/path/to/plugin1.so", "/path/to/plugin2.so"};
        EXPECT_EQ(returned_paths, expected_paths);
    }
}

TEST(Engine_plugin_resource_manager, get_workspace_size_null_engine_config)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    Mock_graph_descriptor mock_graph_desc;

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        EXPECT_HIPDNN_EXCEPTION(resource_manager.get_workspace_size(100, nullptr, &mock_graph_desc),
                                HIPDNN_STATUS_BAD_PARAM);
    }
}

TEST(Engine_plugin_resource_manager, get_workspace_size_invalid_engine_id)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    Mock_graph_descriptor mock_graph_desc;
    hipdnnPluginConstData_t fake_engine_config = {
        reinterpret_cast<const void*>("fake_config"),
        11 // length of "fake_config"
    };

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);

        EXPECT_HIPDNN_EXCEPTION(
            resource_manager.get_workspace_size(999999, &fake_engine_config, &mock_graph_desc),
            HIPDNN_STATUS_INTERNAL_ERROR);
    }
}

TEST(Engine_plugin_resource_manager, set_plugin_paths_with_active_resource_manager)
{
    std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
    std::vector<std::shared_ptr<Engine_plugin>> plugins{mock_plugin};
    std::shared_ptr<Mock_engine_plugin_manager> plugin_manager
        = std::make_shared<Mock_engine_plugin_manager>();

    EXPECT_CALL(*plugin_manager, get_plugins()).WillOnce(::testing::ReturnRef(plugins));
    EXPECT_CALL(*mock_plugin, create_handle())
        .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));
    EXPECT_CALL(*mock_plugin, get_all_engine_ids())
        .WillOnce(::testing::Return(std::vector<int64_t>{100, 101, 102}));
    EXPECT_CALL(*mock_plugin, destroy_handle(testing::Eq(hipdnnEnginePluginHandle_t(0xdeadbeef))));

    {
        Engine_plugin_resource_manager resource_manager(plugin_manager);
        std::vector<std::filesystem::path> plugin_paths = {"/test/path"};

        EXPECT_NO_THROW(Engine_plugin_resource_manager::set_plugin_paths(
            plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

        auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();
        std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());
        EXPECT_EQ(retrieved_paths, expected_paths);

        std::vector<std::filesystem::path> empty_paths;
        EXPECT_NO_THROW(Engine_plugin_resource_manager::set_plugin_paths(
            empty_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
    }
}
