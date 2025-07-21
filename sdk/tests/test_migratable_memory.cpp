// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/migratable_memory.hpp>

using namespace hipdnn_sdk::utilities;

template<typename T>
void init_buffer(T * buffer, size_t size, T mult = 1)
{
    for (size_t i = 0; i < size; ++i) {
        buffer[i] = static_cast<T>(i) * mult;
    }
}

template<typename T>
void check_buffer(const T * buffer, size_t size, T mult = 1)
{
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(buffer[i], static_cast<T>(i) * mult);
    }
}

TEST(MigratableMemory, NotInitialized)
{
    Migratable_memory<float> memory;

    EXPECT_TRUE(memory.empty());
    EXPECT_EQ(memory.size(), 0);
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::NONE);
    EXPECT_EQ(memory.host_data(), nullptr);
}

TEST(MigratableMemory, InitializeWithSize)
{
    Migratable_memory<float> memory(10);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.size(), 10);
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::HOST);
    EXPECT_NE(memory.host_data(), nullptr);
}

TEST(MigratableMemory, MoveConstructor)
{
    Migratable_memory<float> memory1(10);
    float * old_host_data = memory1.host_data();

    Migratable_memory<float> memory2(std::move(memory1));

    EXPECT_TRUE(memory1.empty());
    EXPECT_EQ(memory1.size(), 0);
    EXPECT_EQ(memory1.location(), Migratable_memory<float>::Location::NONE);
    EXPECT_EQ(memory1.host_data(), nullptr);

    EXPECT_FALSE(memory2.empty());
    EXPECT_EQ(memory2.size(), 10);
    EXPECT_EQ(memory2.location(), Migratable_memory<float>::Location::HOST);
    EXPECT_NE(memory2.host_data(), nullptr);
    EXPECT_EQ(memory2.host_data(), old_host_data);
}

TEST(MigratableMemory, MoveAssignment)
{
    Migratable_memory<float> memory1(10);
    float * old_host_data = memory1.host_data();

    Migratable_memory<float> memory2;
    memory2 = std::move(memory1);

    EXPECT_TRUE(memory1.empty());
    EXPECT_EQ(memory1.size(), 0);
    EXPECT_EQ(memory1.location(), Migratable_memory<float>::Location::NONE);
    EXPECT_EQ(memory1.host_data(), nullptr);

    EXPECT_FALSE(memory2.empty());
    EXPECT_EQ(memory2.size(), 10);
    EXPECT_EQ(memory2.location(), Migratable_memory<float>::Location::HOST);
    EXPECT_NE(memory2.host_data(), nullptr);
    EXPECT_EQ(memory2.host_data(), old_host_data);
}

TEST(MigratableMemory, Resize)
{
    Migratable_memory<float> memory(10);
    memory.resize(20);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.size(), 20);
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::HOST);
    EXPECT_NE(memory.host_data(), nullptr);
}

TEST(MigratableMemory, MigrateToDevice)
{
    Migratable_memory<float> memory(10);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.size(), 10);
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::HOST);

    init_buffer(memory.host_data(), memory.size());

    EXPECT_NE(memory.device_data(), nullptr);
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::BOTH);

    check_buffer(memory.device_data(), memory.size());
}

TEST(MigratableMemory, MigrateToHost)
{
    Migratable_memory<float> memory(10);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.size(), 10);
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::HOST);

    init_buffer(memory.host_data(), memory.size());

    check_buffer(memory.device_data(), memory.size());
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::BOTH);

    float array[10];
    init_buffer(array, 10, 2.0f);
    hipError_t err = hipMemcpy(memory.device_data(), array, memory.size() * sizeof(float), hipMemcpyHostToDevice);
    EXPECT_EQ(err, hipSuccess);
    memory.mark_device_modified();
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::DEVICE);

    check_buffer(memory.host_data(), memory.size(), 2.0f);
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::BOTH);
}

TEST(MigratableMemory, Clear)
{
    Migratable_memory<float> memory(10);
    memory.clear();

    EXPECT_TRUE(memory.empty());
    EXPECT_EQ(memory.size(), 0);
    EXPECT_EQ(memory.location(), Migratable_memory<float>::Location::NONE);
    EXPECT_EQ(memory.host_data(), nullptr);
    EXPECT_EQ(memory.device_data(), nullptr);
}

