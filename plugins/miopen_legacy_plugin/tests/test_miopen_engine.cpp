// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/miopen_engine.hpp"
#include "mocks/mock_solver.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <set>

using namespace miopen_legacy_plugin;

TEST(Miopen_engineTest, ConstructorAndId)
{
    std::set<std::unique_ptr<Solver>> solvers;
    Miopen_engine engine(42, solvers);
    EXPECT_EQ(engine.id(),
              42); // /data/hipDNN/plugins/miopen_legacy_plugin/engines/miopen_engine.cpp:13
}

TEST(Miopen_engineTest, WorkspaceSize)
{
    std::set<std::unique_ptr<Solver>> solvers;
    Miopen_engine engine(1, solvers);
    EXPECT_EQ(engine.get_workspace_size(),
              1337); // /data/hipDNN/plugins/miopen_legacy_plugin/engines/miopen_engine.cpp:27
}

TEST(Miopen_engineTest, IsApplicableAlwaysTrue)
{
    std::set<std::unique_ptr<Solver>> solvers;
    Miopen_engine engine(1, solvers);
    EXPECT_TRUE(engine.is_applicable(
        nullptr)); // /data/hipDNN/plugins/miopen_legacy_plugin/engines/miopen_engine.cpp:21
}

TEST(Miopen_engineTest, SolversSetIsEmptyInitially)
{
    std::set<std::unique_ptr<Solver>> solvers;
    Miopen_engine engine(1, solvers);
    EXPECT_TRUE(solvers.empty());
}

TEST(Miopen_engineTest, SolversSetWithMockSolver)
{
    std::set<std::unique_ptr<Solver>> solvers;
    solvers.insert(std::make_unique<Mock_solver>());
    Miopen_engine engine(1, solvers);
    EXPECT_EQ(engine.id(), 1);
    EXPECT_EQ(engine.get_workspace_size(), 1337);
}
