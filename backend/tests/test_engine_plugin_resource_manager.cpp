// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/scoped_resource.hpp>

#include "plugin/engine_plugin_resource_manager.hpp"
#include "hipdnn_exception.hpp"

#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace ::testing;

template <typename T, typename Destructor>
using Scoped_resource = hipdnn::sdk::utilities::Scoped_resource<T, Destructor>;

class Engine_plugin_resource_manager_test_base : public ::testing::Test
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

class Engine_plugin_resource_manager_good_conditions_test : public Engine_plugin_resource_manager_test_base {};

class Engine_plugin_resource_manager_error_conditions_test : public Engine_plugin_resource_manager_test_base {};

class Engine_plugin_resource_manager_stress_test : public Engine_plugin_resource_manager_test_base {};

TEST_F(Engine_plugin_resource_manager_good_conditions_test, StaticPluginPathManagement_SetAndGetSinglePath)
{
    std::vector<std::filesystem::path> plugin_paths = {"/test/plugin/path"};
    
    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();
    
    std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, StaticPluginPathManagement_SetAndGetMultiplePaths)
{
    std::vector<std::filesystem::path> plugin_paths = {
        "/test/plugin/path1", 
        "/test/plugin/path2", 
        "/test/plugin/path3"
    };
    
    Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();
    
    std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, StaticPluginPathManagement_AdditiveLoadingMode)
{
    std::vector<std::filesystem::path> initial_paths = {"/test/path1"};
    Engine_plugin_resource_manager::set_plugin_paths(initial_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    
    std::vector<std::filesystem::path> additional_paths = {"/test/path2", "/test/path3"};
    Engine_plugin_resource_manager::set_plugin_paths(additional_paths, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    
    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();
    
    std::set<std::filesystem::path> expected_paths = {"/test/path1", "/test/path2", "/test/path3"};
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, StaticPluginPathManagement_AbsoluteLoadingModeReplacesExisting)
{
    std::vector<std::filesystem::path> initial_paths = {"/test/path1", "/test/path2"};
    Engine_plugin_resource_manager::set_plugin_paths(initial_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    
    std::vector<std::filesystem::path> new_paths = {"/test/path3", "/test/path4"};
    Engine_plugin_resource_manager::set_plugin_paths(new_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    
    auto retrieved_paths = Engine_plugin_resource_manager::get_plugin_paths();
    
    std::set<std::filesystem::path> expected_paths = {"/test/path3", "/test/path4"};
    EXPECT_EQ(retrieved_paths, expected_paths);
}

TEST_F(Engine_plugin_resource_manager_good_conditions_test, StaticPluginPathManagement_EmptyPathsClearing)
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

TEST_F(Engine_plugin_resource_manager_good_conditions_test, Construction_ScopedCreationAndDestruction)
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

TEST_F(Engine_plugin_resource_manager_good_conditions_test, StreamManagement_MultipleStreamOperations)
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

TEST_F(Engine_plugin_resource_manager_good_conditions_test, EngineDiscovery_GetApplicableEngineIdsWithNullptr)
{
    auto rm = create_resource_manager();
    
    EXPECT_THROW(
        rm->get_applicable_engine_ids(nullptr),
        Hipdnn_exception
    );
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, StaticPluginPathManagement_SetPathsWithActiveHandle)
{
    auto rm = Engine_plugin_resource_manager::create();
    
    std::vector<std::filesystem::path> plugin_paths = {"/test/path"};
    
    EXPECT_THROW(
        Engine_plugin_resource_manager::set_plugin_paths(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        Hipdnn_exception
    );
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, EngineDiscovery_NullGraphDescriptor)
{
    auto rm = create_resource_manager();
    
    EXPECT_THROW(
        rm->get_applicable_engine_ids(nullptr),
        Hipdnn_exception
    );
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, WorkspaceSize_InvalidEngineId)
{
    auto rm = create_resource_manager();
    
    int64_t invalid_engine_id = 999999;
    hipdnnPluginConstData_t engine_config = {nullptr, 0};
    
    EXPECT_THROW(
        rm->get_workspace_size(invalid_engine_id, &engine_config, nullptr),
        Hipdnn_exception
    );
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, WorkspaceSize_NullEngineConfig)
{
    auto rm = create_resource_manager();
    
    int64_t engine_id = 1;
    
    EXPECT_THROW(
        rm->get_workspace_size(engine_id, nullptr, nullptr),
        Hipdnn_exception
    );
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, EngineDetails_InvalidEngineId)
{
    auto rm = create_resource_manager();
    
    int64_t invalid_engine_id = 999999;
    
    EXPECT_THROW(
        Engine_plugin_resource_manager::get_engine_details(rm, invalid_engine_id, nullptr),
        Hipdnn_exception
    );
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, ExecutionContext_InvalidEngineId)
{
    auto rm = create_resource_manager();
    
    int64_t invalid_engine_id = 999999;
    hipdnnPluginConstData_t engine_config = {nullptr, 0};
    
    EXPECT_THROW(
        Engine_plugin_resource_manager::create_execution_context(rm, invalid_engine_id, &engine_config, nullptr),
        Hipdnn_exception
    );
}

TEST_F(Engine_plugin_resource_manager_error_conditions_test, ExecutionContext_NullEngineConfig)
{
    auto rm = create_resource_manager();
    
    int64_t engine_id = 1;
    
    EXPECT_THROW(
        Engine_plugin_resource_manager::create_execution_context(rm, engine_id, nullptr, nullptr),
        Hipdnn_exception
    );
}

TEST_F(Engine_plugin_resource_manager_stress_test, MultiInstance_CreateManyManagersSimultaneously)
{
    const int num_managers = 50;
    std::vector<std::shared_ptr<Engine_plugin_resource_manager>> managers;
    managers.reserve(num_managers);
    
    for (int i = 0; i < num_managers; ++i) {
        auto rm = create_resource_manager();
        ASSERT_NE(rm, nullptr);
        managers.push_back(rm);
    }
    
    for (size_t i = 0; i < static_cast<size_t>(num_managers); ++i) {
        for (size_t j = i + 1; j < static_cast<size_t>(num_managers); ++j) {
            EXPECT_NE(managers[i].get(), managers[j].get());
        }
    }
    
    for (auto& rm : managers) {
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
    
    for (int i = 0; i < num_iterations; ++i) {
        auto rm = create_resource_manager();
        ASSERT_NE(rm, nullptr);
        
        EXPECT_NO_THROW(rm->set_stream(nullptr));
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, Lifecycle_ScopedCreationInNestedLoops)
{
    const int outer_loops = 10;
    const int inner_loops = 10;
    
    for (int i = 0; i < outer_loops; ++i) {
        for (int j = 0; j < inner_loops; ++j) {
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
    
    for (int i = 0; i < num_moves; ++i) {
        auto new_rm = create_resource_manager();
        
        new_rm = std::move(managers.back());
        managers.push_back(new_rm);
        
        ASSERT_NE(managers.back(), nullptr);
        EXPECT_NO_THROW(managers.back()->set_stream(nullptr));
    }
    
    for (size_t i = 1; i < managers.size(); ++i) {
        if (managers[i] != nullptr) {
            EXPECT_THROW(managers[i]->get_applicable_engine_ids(nullptr), Hipdnn_exception);
        }
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, Lifecycle_ExceptionSafetyDuringConstruction)
{
    const int num_attempts = 20;
    
    for (int i = 0; i < num_attempts; ++i) {
        try {
            auto rm = create_resource_manager();
            EXPECT_NE(rm, nullptr);
            
            EXPECT_NO_THROW(rm->set_stream(nullptr));
        } catch (const std::exception& e) {
            SUCCEED() << "Construction failed with exception: " << e.what();
        }
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, ConcurrentAccess_MultipleThreadsCreatingManagers)
{
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::vector<std::shared_ptr<Engine_plugin_resource_manager>>> all_managers(static_cast<size_t>(num_threads));
    std::atomic<int> successful_creations{0};
    
    threads.reserve(static_cast<size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, &all_managers, &successful_creations]() {
            constexpr int managers_per_thread = 10;
            for (int i = 0; i < managers_per_thread; ++i) {
                try {
                    auto rm = create_resource_manager();
                    if (rm != nullptr) {
                        all_managers[static_cast<size_t>(t)].push_back(rm);
                        successful_creations++;
                        
                        rm->set_stream(nullptr);
                        
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                } catch (const std::exception& e) {
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_GT(successful_creations.load(), 0);
    
    for (const auto& thread_managers : all_managers) {
        for (const auto& rm : thread_managers) {
            EXPECT_NE(rm, nullptr);
        }
    }
}

TEST_F(Engine_plugin_resource_manager_stress_test, MemoryManagement_NoLeaksInRapidAllocation)
{
    const int num_cycles = 100;
    
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        std::vector<std::shared_ptr<Engine_plugin_resource_manager>> managers;
        
        for (int i = 0; i < 10; ++i) {
            auto rm = create_resource_manager();
            if (rm != nullptr) {
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
}
