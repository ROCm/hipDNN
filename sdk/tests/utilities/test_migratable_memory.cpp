// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/migratable_memory.hpp>

using namespace hipdnn_sdk::utilities;

template <typename T>
void init_buffer(T* buffer, size_t size, T mult = 1)
{
    for(size_t i = 0; i < size; ++i)
    {
        buffer[i] = static_cast<T>(i) * mult;
    }
}

template <typename T>
void check_buffer(const T* buffer, size_t size, T mult = 1)
{
    for(size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(buffer[i], static_cast<T>(i) * mult);
    }
}

TEST(MigratableMemory, NotInitialized)
{
    Migratable_memory memory;

    EXPECT_TRUE(memory.empty());
    EXPECT_EQ(memory.count(), 0);
    EXPECT_EQ(memory.location(), Migratable_memory::Location::NONE);
    EXPECT_EQ(memory.host_data<float>(), nullptr);
}

TEST(MigratableMemory, InitializeWithSize)
{
    Migratable_memory memory(10, sizeof(float));

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Migratable_memory::Location::HOST);
    EXPECT_NE(memory.host_data<float>(), nullptr);
}

TEST(MigratableMemory, MoveConstructor)
{
    Migratable_memory memory1(10, sizeof(float));
    auto* old_host_data = memory1.host_data<float>();

    Migratable_memory memory2(std::move(memory1));

    EXPECT_TRUE(memory1.empty());
    EXPECT_EQ(memory1.count(), 0);
    EXPECT_EQ(memory1.location(), Migratable_memory::Location::NONE);
    EXPECT_EQ(memory1.host_data<float>(), nullptr);

    EXPECT_FALSE(memory2.empty());
    EXPECT_EQ(memory2.count(), 10);
    EXPECT_EQ(memory2.location(), Migratable_memory::Location::HOST);
    EXPECT_NE(memory2.host_data<float>(), nullptr);
    EXPECT_EQ(memory2.host_data<float>(), old_host_data);
}

TEST(MigratableMemory, MoveAssignment)
{
    Migratable_memory memory1(10, sizeof(float));
    auto* old_host_data = memory1.host_data<float>();

    Migratable_memory memory2;
    memory2 = std::move(memory1);

    EXPECT_TRUE(memory1.empty());
    EXPECT_EQ(memory1.count(), 0);
    EXPECT_EQ(memory1.location(), Migratable_memory::Location::NONE);
    EXPECT_EQ(memory1.host_data<float>(), nullptr);

    EXPECT_FALSE(memory2.empty());
    EXPECT_EQ(memory2.count(), 10);
    EXPECT_EQ(memory2.location(), Migratable_memory::Location::HOST);
    EXPECT_NE(memory2.host_data<float>(), nullptr);
    EXPECT_EQ(memory2.host_data<float>(), old_host_data);
}

TEST(MigratableMemory, Resize)
{
    Migratable_memory memory(10, sizeof(float));
    memory.resize(20);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 20);
    EXPECT_EQ(memory.location(), Migratable_memory::Location::HOST);
    EXPECT_NE(memory.host_data<float>(), nullptr);
}

TEST(MigratableMemory, MigrateToDevice)
{
    SKIP_IF_NO_DEVICES();

    Migratable_memory memory(10, sizeof(float));

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Migratable_memory::Location::HOST);

    init_buffer(memory.host_data<float>(), memory.count());

    EXPECT_NE(memory.device_data<float>(), nullptr);
    EXPECT_EQ(memory.location(), Migratable_memory::Location::BOTH);

    check_buffer(memory.device_data<float>(), memory.count());
}

TEST(MigratableMemory, MigrateToHost)
{
    SKIP_IF_NO_DEVICES();

    Migratable_memory memory(10, sizeof(float));

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Migratable_memory::Location::HOST);

    init_buffer(memory.host_data<float>(), memory.count());

    check_buffer(memory.device_data<float>(), memory.count());
    EXPECT_EQ(memory.location(), Migratable_memory::Location::BOTH);

    std::array<float, 10> array;
    init_buffer(array.data(), 10, 2.0f);
    hipError_t err = hipMemcpy(memory.device_data<float>(),
                               array.data(),
                               memory.count() * sizeof(float),
                               hipMemcpyHostToDevice);
    EXPECT_EQ(err, hipSuccess);
    memory.mark_device_modified();
    EXPECT_EQ(memory.location(), Migratable_memory::Location::DEVICE);

    check_buffer(memory.host_data<float>(), memory.count(), 2.0f);
    EXPECT_EQ(memory.location(), Migratable_memory::Location::BOTH);
}

TEST(MigratableMemory, Clear)
{
    SKIP_IF_NO_DEVICES();

    Migratable_memory memory(10, sizeof(float));
    memory.clear();

    EXPECT_TRUE(memory.empty());
    EXPECT_EQ(memory.count(), 0);
    EXPECT_EQ(memory.location(), Migratable_memory::Location::NONE);
    EXPECT_EQ(memory.host_data<float>(), nullptr);
    EXPECT_EQ(memory.device_data<float>(), nullptr);
}
