// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// #include <atomic>
// #include <filesystem>
// #include <functional>
// #include <memory>
// #include <numeric> // for std::iota
// #include <thread>
// #include <vector>

#include <gtest/gtest.h>
// #include <hip/hip_runtime.h>
// #include <hipdnn_sdk/test_utilities/test_utilities.hpp>
// #include <hipdnn_sdk/utilities/scoped_resource.hpp>

// #include "descriptors/engine_config_descriptor.hpp"
// #include "descriptors/execution_plan_descriptor.hpp"
// #include "descriptors/graph_descriptor.hpp"
// #include "descriptors/variant_descriptor.hpp"
// #include "handle/handle.hpp"
// #include "hipdnn_sdk/data_objects/engine_details_generated.h"
#include "plugin/engine_plugin_resource_manager.hpp"
// #include "test_utilities/engine_plugin_test_helpers.hpp"
#include "plugins/mocks/mock_engine_plugin.hpp"
#include "plugins/mocks/mock_engine_plugin_manager.hpp"

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace ::testing;
//using namespace test_utilities;

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

// TEST(Engine_plugin_resource_manager, set_stream)
// {
//     std::shared_ptr<Mock_engine_plugin> mock_plugin = std::make_shared<Mock_engine_plugin>();
//     std::vector<std::shared_ptr<Engine_plugin>> plugins { mock_plugin };

//     std::shared_ptr<Mock_engine_plugin_manager> plugin_manager = std::make_shared<Mock_engine_plugin_manager>();

//     EXPECT_CALL(*plugin_manager, get_plugins())
//         .WillOnce(::testing::ReturnRef(plugins));

//     EXPECT_CALL(*mock_plugin, create_handle())
//         .WillOnce(::testing::Return(hipdnnEnginePluginHandle_t(0xdeadbeef)));

//     EXPECT_CALL(*mock_plugin, set_stream(
//         hipdnnEnginePluginHandle_t(0xdeadbeef),
//         hipStream_t(0x12345678)
//     ));

//     EXPECT_CALL(*mock_plugin, destroy_handle(
//         hipdnnEnginePluginHandle_t(0xdeadbeef)
//     ));

//     {
//         Engine_plugin_resource_manager resource_manager(plugin_manager);

//         resource_manager.set_stream(hipStream_t(0x12345678));
//     }
// }

/*

template <typename T, typename Destructor>
using Scoped_resource = hipdnn::sdk::utilities::Scoped_resource<T, Destructor>;

class Engine_plugin_loading_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        Engine_plugin_resource_manager::set_plugin_paths({}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    }

    void TearDown() override
    {
        Engine_plugin_resource_manager::set_plugin_paths({}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    }

    static std::shared_ptr<Engine_plugin_resource_manager> create_resource_manager()
    {
        return Engine_plugin_resource_manager::create();
    }
};

class Engine_plugin_resource_manager_good_conditions_test
    : public Engine_plugin_resource_manager_test_base
{
};

class Engine_plugin_resource_manager_error_conditions_test
    : public Engine_plugin_resource_manager_test_base
{
};

class Engine_plugin_resource_manager_stress_test : public Engine_plugin_resource_manager_test_base
{
};

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       StaticPluginPathManagement_SetAndGetSinglePath)
{
    std::vector<std::filesystem::path> plugin_paths = {"/test/plugin/path"};

    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();

    std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       StaticPluginPathManagement_SetAndGetMultiplePaths)
{
    std::vector<std::filesystem::path> plugin_paths
        = {"/test/plugin/path1", "/test/plugin/path2", "/test/plugin/path3"};

    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();

    std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       StaticPluginPathManagement_AdditiveLoadingMode)
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

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       StaticPluginPathManagement_AbsoluteLoadingModeReplacesExisting)
{
    std::vector<std::filesystem::path> initial_paths = {"/test/path1", "/test/path2"};
    Engine_plugin_resource_manager::set_plugin_paths(initial_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    std::vector<std::filesystem::path> new_paths = {"/test/path3", "/test/path4"};
    Engine_plugin_resource_manager::set_plugin_paths(new_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();

    std::set<std::filesystem::path> expected_paths = {"/test/path3", "/test/path4"};
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       StaticPluginPathManagement_EmptyPathsClearing)
{
    std::vector<std::filesystem::path> plugin_paths = {"/test/path1", "/test/path2"};
    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    std::vector<std::filesystem::path> empty_paths;
    Engine_plugin_resource_manager::set_plugin_paths(empty_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();
    EXPECT_TRUE(retrieved_paths.empty());
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, Construction_CreateSingleInstance)
{
    auto resource_manager = create_resource_manager();
    ASSERT_NE(resource_manager, nullptr);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, Construction_CreateMultipleInstances)
{
    auto rm1 = create_resource_manager();
    auto rm2 = create_resource_manager();
    auto rm3 = create_resource_manager();

    ASSERT_NE(rm1, nullptr);
    ASSERT_NE(rm2, nullptr);
    ASSERT_NE(rm3, nullptr);

    EXPECT_NE(rm1.get(), rm2.get());
    EXPECT_NE(rm2.get(), rm3.get());
    EXPECT_NE(rm1.get(), rm3.get());
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, Construction_MoveConstructor)
{
    auto rm1 = create_resource_manager();
    auto original_ptr = rm1.get();

    auto rm2 = std::move(rm1);

    EXPECT_EQ(rm2.get(), original_ptr);
    EXPECT_EQ(rm1.get(), nullptr);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, Construction_MoveAssignment)
{
    auto rm1 = create_resource_manager();
    auto rm2 = create_resource_manager();
    auto original_ptr = rm1.get();

    rm2 = std::move(rm1);

    EXPECT_EQ(rm2.get(), original_ptr);
    EXPECT_EQ(rm1.get(), nullptr);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       Construction_ScopedCreationAndDestruction)
{
    {
        auto rm = create_resource_manager();
        ASSERT_NE(rm, nullptr);
    }
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, StreamManagement_SetValidStream)
{
    SKIP_IF_NO_DEVICES();

    auto rm = create_resource_manager();

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    Scoped_resource stream_res(stream, [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });

    EXPECT_NO_THROW(rm->set_stream(stream));
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, StreamManagement_SetNullStream)
{
    auto rm = create_resource_manager();

    EXPECT_NO_THROW(rm->set_stream(nullptr));
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       StreamManagement_MultipleStreamOperations)
{
    SKIP_IF_NO_DEVICES();

    auto rm = create_resource_manager();

    hipStream_t stream1;
    hipStream_t stream2;
    ASSERT_EQ(hipStreamCreate(&stream1), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream2), hipSuccess);

    Scoped_resource stream1_res(stream1, [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });
    Scoped_resource stream2_res(stream2, [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });

    EXPECT_NO_THROW(rm->set_stream(stream1));
    EXPECT_NO_THROW(rm->set_stream(stream2));
    EXPECT_NO_THROW(rm->set_stream(nullptr));
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       EngineDiscovery_GetApplicableEngineIdsWithNullptr)
{
    auto rm = create_resource_manager();

    EXPECT_THROW(rm->get_applicable_engine_ids(nullptr), Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       EngineDiscovery_GetApplicableEngineIdsWithLoadedPlugin)
{
    std::vector<std::filesystem::path> plugin_paths = {"./hipdnn_test_engine_plugin1"};
    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto rm = create_resource_manager();

    try
    {
        hipdnnHandle_t test_handle;
        ASSERT_EQ(hipdnnCreate(&test_handle), HIPDNN_STATUS_SUCCESS);
        Scoped_resource handle_res(test_handle,
                                   [](hipdnnHandle_t h) { std::ignore = hipdnnDestroy(h); });

        auto graph_desc = std::make_unique<Graph_descriptor>();
        graph_desc->finalize();

        auto engine_ids = rm->get_applicable_engine_ids(graph_desc.get());

        EXPECT_EQ(engine_ids.size(), 3);
        EXPECT_EQ(engine_ids[0], 100);
        EXPECT_EQ(engine_ids[1], 101);
        EXPECT_EQ(engine_ids[2], 102);
    }
    catch(const Hipdnn_exception& e)
    {
        SUCCEED() << "Plugin loading or engine discovery failed as expected: " << e.what();
    }
    catch(const std::exception& e)
    {
        SUCCEED() << "Plugin loading or engine discovery failed: " << e.what();
    }
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test,
       IntegrationWorkflow_HandleCreationToPluginExecution)
{
    std::vector<std::filesystem::path> plugin_paths = {"./hipdnn_test_engine_plugin1"};
    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();
    std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());
    EXPECT_EQ(retrieved_paths, expected_paths);

    try
    {
        hipdnnHandle_t test_handle;
        ASSERT_EQ(hipdnnCreate(&test_handle), HIPDNN_STATUS_SUCCESS);
        Scoped_resource handle_res(test_handle,
                                   [](hipdnnHandle_t h) { std::ignore = hipdnnDestroy(h); });

        auto rm = create_resource_manager();
        ASSERT_NE(rm, nullptr);

        hipStream_t stream;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        Scoped_resource stream_res(stream,
                                   [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });
        rm->set_stream(stream);

        auto graph_desc = std::make_unique<Graph_descriptor>();
        graph_desc->finalize();

        auto engine_ids = rm->get_applicable_engine_ids(graph_desc.get());
        EXPECT_EQ(engine_ids.size(), 3);
        EXPECT_EQ(engine_ids[0], 100);
        EXPECT_EQ(engine_ids[1], 101);
        EXPECT_EQ(engine_ids[2], 102);

        auto first_engine_id = engine_ids[0];
        hipdnnPluginConstData_t engine_config = {nullptr, 0};
        auto workspace_size
            = rm->get_workspace_size(first_engine_id, &engine_config, graph_desc.get());
        EXPECT_EQ(workspace_size, 4096);

        auto engine_details = Engine_plugin_resource_manager::get_engine_details(
            rm, first_engine_id, graph_desc.get());
        ASSERT_NE(engine_details, nullptr);
        EXPECT_EQ(engine_details->get()->engine_id(), first_engine_id);

        auto execution_context = Engine_plugin_resource_manager::create_execution_context(
            rm, first_engine_id, &engine_config, graph_desc.get());
        ASSERT_NE(execution_context, nullptr);
        ASSERT_NE(execution_context->get(), nullptr);

        for(auto engine_id : engine_ids)
        {
            auto details = Engine_plugin_resource_manager::get_engine_details(
                rm, engine_id, graph_desc.get());
            ASSERT_NE(details, nullptr);
            EXPECT_EQ(details->get()->engine_id(), engine_id);

            auto ws_size = rm->get_workspace_size(engine_id, &engine_config, graph_desc.get());
            EXPECT_GT(ws_size, 0);

            auto exec_ctx = Engine_plugin_resource_manager::create_execution_context(
                rm, engine_id, &engine_config, graph_desc.get());
            ASSERT_NE(exec_ctx, nullptr);
        }
    }
    catch(const Hipdnn_exception& e)
    {
        SUCCEED() << "Integration workflow failed as expected (plugin may not be available): "
                  << e.what();
    }
    catch(const std::exception& e)
    {
        SUCCEED() << "Integration workflow failed: " << e.what();
    }
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test,
       StaticPluginPathManagement_SetPathsWithActiveHandle)
{
    auto rm = Engine_plugin_resource_manager::create();

    std::vector<std::filesystem::path> plugin_paths = {"/test/path"};

    EXPECT_THROW(Engine_plugin_resource_manager::set_plugin_paths(plugin_paths,
                                                                  HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                 Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, EngineDiscovery_NullGraphDescriptor)
{
    auto rm = create_resource_manager();

    EXPECT_THROW(rm->get_applicable_engine_ids(nullptr), Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, WorkspaceSize_InvalidEngineId)
{
    auto rm = create_resource_manager();

    int64_t invalid_engine_id = 999999;
    hipdnnPluginConstData_t engine_config = {nullptr, 0};

    EXPECT_THROW(rm->get_workspace_size(invalid_engine_id, &engine_config, nullptr),
                 Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, WorkspaceSize_NullEngineConfig)
{
    auto rm = create_resource_manager();

    int64_t engine_id = 1;

    EXPECT_THROW(rm->get_workspace_size(engine_id, nullptr, nullptr), Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, EngineDetails_InvalidEngineId)
{
    auto rm = create_resource_manager();

    int64_t invalid_engine_id = 999999;

    EXPECT_THROW(Engine_plugin_resource_manager::get_engine_details(rm, invalid_engine_id, nullptr),
                 Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, ExecutionContext_InvalidEngineId)
{
    auto rm = create_resource_manager();

    int64_t invalid_engine_id = 999999;
    hipdnnPluginConstData_t engine_config = {nullptr, 0};

    EXPECT_THROW(Engine_plugin_resource_manager::create_execution_context(
                     rm, invalid_engine_id, &engine_config, nullptr),
                 Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, ExecutionContext_NullEngineConfig)
{
    auto rm = create_resource_manager();

    int64_t engine_id = 1;

    EXPECT_THROW(
        Engine_plugin_resource_manager::create_execution_context(rm, engine_id, nullptr, nullptr),
        Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_stress_test, MultiInstance_CreateManyManagersSimultaneously)
{
    const int num_managers = 50;
    std::vector<std::shared_ptr<Engine_plugin_resource_manager>> managers;
    managers.reserve(num_managers);

    for(int i = 0; i < num_managers; ++i)
    {
        auto rm = create_resource_manager();
        ASSERT_NE(rm, nullptr);
        managers.push_back(rm);
    }

    for(size_t i = 0; i < static_cast<size_t>(num_managers); ++i)
    {
        for(size_t j = i + 1; j < static_cast<size_t>(num_managers); ++j)
        {
            EXPECT_NE(managers[i].get(), managers[j].get());
        }
    }

    for(auto& rm : managers)
    {
        EXPECT_NO_THROW(rm->set_stream(nullptr));
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, MultiInstance_ResourceIsolation)
{
    auto rm1 = create_resource_manager();
    auto rm2 = create_resource_manager();

    EXPECT_NO_THROW(rm1->set_stream(nullptr));
    EXPECT_NO_THROW(rm2->set_stream(nullptr));

    EXPECT_THROW(rm1->get_applicable_engine_ids(nullptr), Hipdnn_exception);
    EXPECT_THROW(rm2->get_applicable_engine_ids(nullptr), Hipdnn_exception);
}

TEST_F(Engine_plugin_resource_manager_stress_test, Lifecycle_RapidCreateDestroyLoop)
{
    const int num_iterations = 100;

    for(int i = 0; i < num_iterations; ++i)
    {
        auto rm = create_resource_manager();
        ASSERT_NE(rm, nullptr);

        EXPECT_NO_THROW(rm->set_stream(nullptr));
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, Lifecycle_ScopedCreationInNestedLoops)
{
    const int outer_loops = 10;
    const int inner_loops = 10;

    for(int i = 0; i < outer_loops; ++i)
    {
        for(int j = 0; j < inner_loops; ++j)
        {
            auto rm = create_resource_manager();
            ASSERT_NE(rm, nullptr);

            auto rm_moved = std::move(rm);
            ASSERT_NE(rm_moved, nullptr);
            EXPECT_EQ(rm, nullptr);

            EXPECT_NO_THROW(rm_moved->set_stream(nullptr));
        }
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, Lifecycle_MoveOperationsUnderStress)
{
    const int num_moves = 50;
    std::vector<std::shared_ptr<Engine_plugin_resource_manager>> managers;

    auto rm = create_resource_manager();
    managers.push_back(rm);

    for(int i = 0; i < num_moves; ++i)
    {
        auto new_rm = create_resource_manager();

        new_rm = std::move(managers.back());
        managers.push_back(new_rm);

        ASSERT_NE(managers.back(), nullptr);
        EXPECT_NO_THROW(managers.back()->set_stream(nullptr));
    }

    for(size_t i = 1; i < managers.size(); ++i)
    {
        if(managers[i] != nullptr)
        {
            EXPECT_THROW(managers[i]->get_applicable_engine_ids(nullptr), Hipdnn_exception);
        }
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, Lifecycle_ExceptionSafetyDuringConstruction)
{
    const int num_attempts = 20;

    for(int i = 0; i < num_attempts; ++i)
    {
        try
        {
            auto rm = create_resource_manager();
            EXPECT_NE(rm, nullptr);

            EXPECT_NO_THROW(rm->set_stream(nullptr));
        }
        catch(const std::exception& e)
        {
            SUCCEED() << "Construction failed with exception: " << e.what();
        }
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, ConcurrentAccess_MultipleThreadsCreatingManagers)
{
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::vector<std::shared_ptr<Engine_plugin_resource_manager>>> all_managers(
        static_cast<size_t>(num_threads));
    std::atomic<int> successful_creations{0};

    threads.reserve(static_cast<size_t>(num_threads));
    for(int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([t, &all_managers, &successful_creations]() {
            constexpr int managers_per_thread = 10;
            for(int i = 0; i < managers_per_thread; ++i)
            {
                try
                {
                    auto rm = create_resource_manager();
                    if(rm != nullptr)
                    {
                        all_managers[static_cast<size_t>(t)].push_back(rm);
                        successful_creations++;

                        rm->set_stream(nullptr);

                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                }
                catch(const std::exception& e)
                {
                }
            }
        });
    }

    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_GT(successful_creations.load(), 0);

    for(const auto& thread_managers : all_managers)
    {
        for(const auto& rm : thread_managers)
        {
            EXPECT_NE(rm, nullptr);
        }
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, MemoryManagement_NoLeaksInRapidAllocation)
{
    const int num_cycles = 100;

    for(int cycle = 0; cycle < num_cycles; ++cycle)
    {
        std::vector<std::shared_ptr<Engine_plugin_resource_manager>> managers;

        for(int i = 0; i < 10; ++i)
        {
            auto rm = create_resource_manager();
            if(rm != nullptr)
            {
                managers.push_back(rm);
                rm->set_stream(nullptr);
            }
        }
    }

    SUCCEED();
}

TEST_F(Engine_plugin_resource_manager_stress_test, EdgeCase_SelfMoveAssignment)
{
    auto rm = create_resource_manager();
    auto original_ptr = rm.get();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
    rm = std::move(rm);
#pragma clang diagnostic pop

    EXPECT_EQ(rm.get(), original_ptr);
    EXPECT_NO_THROW(rm->set_stream(nullptr));
}

TEST_F(Engine_plugin_resource_manager_stress_test, EdgeCase_MoveFromMovedObject)
{
    auto rm1 = create_resource_manager();
    auto rm2 = create_resource_manager();

    rm2 = std::move(rm1);
    EXPECT_EQ(rm1.get(), nullptr);

    auto rm3 = std::move(rm1);
    EXPECT_EQ(rm3.get(), nullptr);
    EXPECT_EQ(rm1.get(), nullptr);
}

TEST(GPU_EnginePluginResourceManagerTest, LoadPluginsAndExecuteOpGraph)
{
    SKIP_IF_NO_DEVICES();

    std::vector<std::filesystem::path> plugin_paths = {"./hipdnn_test_engine_plugin1"};

    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();
    std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());

    EXPECT_EQ(retrieved_paths, expected_paths);

    auto resource_manager = Engine_plugin_resource_manager::create();

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    Scoped_resource stream_res(stream, [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });

    resource_manager->set_stream(stream);

    try
    {
        auto engine_ids = resource_manager->get_applicable_engine_ids(nullptr);

        if(!engine_ids.empty())
        {
            auto first_engine_id = engine_ids[0];
            EXPECT_EQ(first_engine_id, 100);

            hipdnnPluginConstData_t engine_config = {nullptr, 0};

            auto workspace_size
                = resource_manager->get_workspace_size(first_engine_id, &engine_config, nullptr);
            EXPECT_EQ(workspace_size, 4096);

            auto execution_context = Engine_plugin_resource_manager::create_execution_context(
                resource_manager, first_engine_id, &engine_config, nullptr);
            EXPECT_NE(execution_context, nullptr);

            const size_t data_size = 512;
            const size_t buffer_size = data_size * sizeof(uint32_t);

            uint32_t* input_device;
            uint32_t* output_device;
            void* workspace_device;

            ASSERT_EQ(hipMalloc(&input_device, buffer_size), hipSuccess);
            ASSERT_EQ(hipMalloc(&output_device, buffer_size), hipSuccess);
            ASSERT_EQ(hipMalloc(&workspace_device, workspace_size), hipSuccess);

            Scoped_resource input_res(input_device, [](uint32_t* p) { std::ignore = hipFree(p); });
            Scoped_resource output_res(output_device,
                                       [](uint32_t* p) { std::ignore = hipFree(p); });
            Scoped_resource workspace_res(workspace_device,
                                          [](void* p) { std::ignore = hipFree(p); });

            std::vector<uint32_t> host_input(data_size);
            std::vector<uint32_t> host_output(data_size, 0);

            for(size_t i = 0; i < data_size; ++i)
            {
                host_input[i] = static_cast<uint32_t>(i + 1);
            }

            ASSERT_EQ(
                hipMemcpy(input_device, host_input.data(), buffer_size, hipMemcpyHostToDevice),
                hipSuccess);
        }
    }
    catch(const Hipdnn_exception& e)
    {
        SUCCEED() << "Plugin loaded but execution failed as expected: " << e.what();
    }
    catch(const std::exception& e)
    {
        SUCCEED() << "Plugin loaded but execution failed: " << e.what();
    }
}
std::filesystem::path Engine_plugin_resource_manager_test::static_test_plugin_dir;
std::filesystem::path Engine_plugin_resource_manager_test::static_test_plugin_path;
bool Engine_plugin_resource_manager_test::plugin_available = false;
std::shared_ptr<plugin::Engine_plugin_resource_manager>
    Engine_plugin_resource_manager_test::static_resource_manager;
*/
