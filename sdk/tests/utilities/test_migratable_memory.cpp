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

template <typename T>
void check_buffer_synchronized(const T* buffer,
                               size_t size,
                               hipStream_t stream = nullptr,
                               T mult = 1)
{
    hipError_t error = hipStreamSynchronize(stream);
    EXPECT_EQ(error, hipSuccess) << "Error synchronizing stream";

    for(size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(buffer[i], static_cast<T>(i) * mult);
    }
}

TEST(MigratableMemory, NotInitialized)
{
    Migratable_memory<float> memory;

    EXPECT_TRUE(memory.empty());
    EXPECT_EQ(memory.count(), 0);
    EXPECT_EQ(memory.location(), Memory_location::NONE);
    EXPECT_EQ(memory.host_data(), nullptr);
}

TEST(MigratableMemory, InitializeWithSize)
{
    Migratable_memory<float> memory(10);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Memory_location::HOST);
    EXPECT_NE(memory.host_data(), nullptr);
}

TEST(MigratableMemory, MoveConstructor)
{
    Migratable_memory<float> memory1(10);
    auto* old_host_data = memory1.host_data();

    Migratable_memory<float> memory2(std::move(memory1));

    EXPECT_TRUE(memory1.empty());
    EXPECT_EQ(memory1.count(), 0);
    EXPECT_EQ(memory1.location(), Memory_location::NONE);
    EXPECT_EQ(memory1.host_data(), nullptr);

    EXPECT_FALSE(memory2.empty());
    EXPECT_EQ(memory2.count(), 10);
    EXPECT_EQ(memory2.location(), Memory_location::HOST);
    EXPECT_NE(memory2.host_data(), nullptr);
    EXPECT_EQ(memory2.host_data(), old_host_data);
}

TEST(MigratableMemory, MoveAssignment)
{
    Migratable_memory<float> memory1(10);
    auto* old_host_data = memory1.host_data();

    Migratable_memory<float> memory2;
    memory2 = std::move(memory1);

    EXPECT_TRUE(memory1.empty());
    EXPECT_EQ(memory1.count(), 0);
    EXPECT_EQ(memory1.location(), Memory_location::NONE);
    EXPECT_EQ(memory1.host_data(), nullptr);

    EXPECT_FALSE(memory2.empty());
    EXPECT_EQ(memory2.count(), 10);
    EXPECT_EQ(memory2.location(), Memory_location::HOST);
    EXPECT_NE(memory2.host_data(), nullptr);
    EXPECT_EQ(memory2.host_data(), old_host_data);
}

TEST(MigratableMemory, Resize)
{
    Migratable_memory<float> memory(10);
    memory.resize(20);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 20);
    EXPECT_EQ(memory.location(), Memory_location::HOST);
    EXPECT_NE(memory.host_data(), nullptr);
}

TEST(MigratableMemory, MigrateToDevice)
{
    SKIP_IF_NO_DEVICES();

    Migratable_memory<float> memory(10);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Memory_location::HOST);

    init_buffer(memory.host_data(), memory.count());

    EXPECT_NE(memory.device_data(), nullptr);
    EXPECT_EQ(memory.location(), Memory_location::BOTH);

    check_buffer(static_cast<float*>(memory.device_data()), memory.count());
}

TEST(MigratableMemory, MigrateToDeviceNonDefaultStream)
{
    SKIP_IF_NO_DEVICES();

    hipStream_t stream;
    hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    EXPECT_EQ(error, hipSuccess) << "Failed to create HIP stream";
    ASSERT_NE(stream, nullptr) << "Failed to create HIP stream";

    Migratable_memory<float> memory(10, stream);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Memory_location::HOST);

    init_buffer(memory.host_data(), memory.count());

    EXPECT_NE(memory.device_data(), nullptr);
    EXPECT_EQ(memory.location(), Memory_location::BOTH);

    check_buffer_synchronized(static_cast<float*>(memory.device_data()), memory.count(), stream);

    error = hipStreamDestroy(stream);
    EXPECT_EQ(error, hipSuccess) << "Failed to destroy HIP stream";
}

TEST(MigratableMemory, MigrateToDeviceAsyncNonDefaultStream)
{
    SKIP_IF_NO_DEVICES();

    hipStream_t stream;
    hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    EXPECT_EQ(error, hipSuccess) << "Failed to create HIP stream";
    ASSERT_NE(stream, nullptr) << "Failed to create HIP stream";

    Migratable_memory<float> memory(10, stream);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Memory_location::HOST);

    init_buffer(memory.host_data(), memory.count());

    EXPECT_NE(memory.device_data(), nullptr);
    EXPECT_EQ(memory.location(), Memory_location::BOTH);

    check_buffer_synchronized(
        static_cast<float*>(memory.device_data_async()), memory.count(), stream);

    error = hipStreamDestroy(stream);
    EXPECT_EQ(error, hipSuccess) << "Failed to destroy HIP stream";
}

TEST(MigratableMemory, MigrateToHost)
{
    SKIP_IF_NO_DEVICES();

    Migratable_memory<float> memory(10);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Memory_location::HOST);

    init_buffer(memory.host_data(), memory.count());

    check_buffer(static_cast<float*>(memory.device_data()), memory.count());
    EXPECT_EQ(memory.location(), Memory_location::BOTH);

    std::array<float, 10> array;
    init_buffer(array.data(), 10, 2.0f);
    hipError_t err = hipMemcpy(
        memory.device_data(), array.data(), memory.count() * sizeof(float), hipMemcpyHostToDevice);
    EXPECT_EQ(err, hipSuccess);
    memory.mark_device_modified();
    EXPECT_EQ(memory.location(), Memory_location::DEVICE);

    check_buffer_synchronized(memory.host_data(), memory.count(), nullptr, 2.0f);
    EXPECT_EQ(memory.location(), Memory_location::BOTH);
}

TEST(MigratableMemory, MigrateToHostNonDefaultStream)
{
    SKIP_IF_NO_DEVICES();

    hipStream_t stream;
    hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    EXPECT_EQ(error, hipSuccess) << "Failed to create HIP stream";
    ASSERT_NE(stream, nullptr) << "Failed to create HIP stream";

    Migratable_memory<float> memory(10, stream);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Memory_location::HOST);

    init_buffer(memory.host_data(), memory.count());

    check_buffer_synchronized(static_cast<float*>(memory.device_data()), memory.count(), stream);
    EXPECT_EQ(memory.location(), Memory_location::BOTH);

    std::array<float, 10> array;
    init_buffer(array.data(), 10, 2.0f);
    hipError_t err = hipMemcpyWithStream(memory.device_data(),
                                         array.data(),
                                         memory.count() * sizeof(float),
                                         hipMemcpyHostToDevice,
                                         stream);
    EXPECT_EQ(err, hipSuccess);
    memory.mark_device_modified();
    EXPECT_EQ(memory.location(), Memory_location::DEVICE);

    check_buffer_synchronized(memory.host_data(), memory.count(), stream, 2.0f);
    EXPECT_EQ(memory.location(), Memory_location::BOTH);

    error = hipStreamDestroy(stream);
    EXPECT_EQ(error, hipSuccess) << "Failed to destroy HIP stream";
}

TEST(MigratableMemory, MigrateToHostAsyncNonDefaultStream)
{
    SKIP_IF_NO_DEVICES();

    hipStream_t stream;
    hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    EXPECT_EQ(error, hipSuccess) << "Failed to create HIP stream";
    ASSERT_NE(stream, nullptr) << "Failed to create HIP stream";

    Migratable_memory<float> memory(10, stream);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), Memory_location::HOST);

    init_buffer(memory.host_data(), memory.count());

    check_buffer_synchronized(
        static_cast<float*>(memory.device_data_async()), memory.count(), stream);
    EXPECT_EQ(memory.location(), Memory_location::BOTH);

    std::array<float, 10> array;
    init_buffer(array.data(), 10, 2.0f);
    hipError_t err = hipMemcpyWithStream(memory.device_data(),
                                         array.data(),
                                         memory.count() * sizeof(float),
                                         hipMemcpyHostToDevice,
                                         stream);
    EXPECT_EQ(err, hipSuccess);
    memory.mark_device_modified();
    EXPECT_EQ(memory.location(), Memory_location::DEVICE);

    check_buffer_synchronized(memory.host_data_async(), memory.count(), stream, 2.0f);
    EXPECT_EQ(memory.location(), Memory_location::BOTH);

    error = hipStreamDestroy(stream);
    EXPECT_EQ(error, hipSuccess) << "Failed to destroy HIP stream";
}

TEST(MigratableMemory, Clear)
{
    SKIP_IF_NO_DEVICES();

    Migratable_memory<float> memory(10);
    memory.clear();

    EXPECT_TRUE(memory.empty());
    EXPECT_EQ(memory.count(), 0);
    EXPECT_EQ(memory.location(), Memory_location::NONE);
    EXPECT_EQ(memory.host_data(), nullptr);
    EXPECT_EQ(memory.device_data(), nullptr);
}
