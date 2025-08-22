// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/allocators.hpp>
#include <list>
#include <memory>
#include <vector>

using namespace hipdnn_sdk::utilities;

// Test that allocators work with STL containers
TEST(Allocators, HostAllocatorWithVector)
{
    std::vector<int, Host_allocator<int>> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);
}

TEST(Allocators, HostAllocatorWithList)
{
    std::list<double, Host_allocator<double>> lst;
    lst.push_back(1.5);
    lst.push_back(2.5);
    lst.push_back(3.5);

    auto it = lst.begin();
    EXPECT_EQ(*it++, 1.5);
    EXPECT_EQ(*it++, 2.5);
    EXPECT_EQ(*it++, 3.5);
}

TEST(Allocators, HostAllocatorBasicOperations)
{
    Host_allocator<int> alloc;

    // Allocate memory for 10 integers
    auto ptr = alloc.allocate(10);
    ASSERT_NE(ptr, nullptr);

    // Construct objects
    for(int i = 0; i < 10; ++i)
    {
        alloc.construct(&ptr[i], i * 2);
    }

    // Verify values
    for(int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(ptr[i], i * 2);
    }

    // Destroy objects
    for(int i = 0; i < 10; ++i)
    {
        alloc.destroy(&ptr[i]);
    }

    // Deallocate memory
    alloc.deallocate(ptr, 10);
}

TEST(Allocators, HostAllocatorRebind)
{
    Host_allocator<int> int_alloc;
    typename Host_allocator<int>::template rebind<double>::other double_alloc;

    auto int_ptr = int_alloc.allocate(5);
    auto double_ptr = double_alloc.allocate(5);

    ASSERT_NE(int_ptr, nullptr);
    ASSERT_NE(double_ptr, nullptr);

    int_alloc.deallocate(int_ptr, 5);
    double_alloc.deallocate(double_ptr, 5);
}

TEST(Allocators, HostAllocatorComparison)
{
    Host_allocator<int> alloc1;
    Host_allocator<int> alloc2;
    Host_allocator<double> alloc3;

    EXPECT_TRUE(alloc1 == alloc2);
    EXPECT_FALSE(alloc1 != alloc2);
    EXPECT_TRUE(alloc1 == alloc3);
}

TEST(Allocators, PinnedHostAllocatorWithVector)
{
    SKIP_IF_NO_DEVICES();

    std::vector<float, Pinned_host_allocator<float>> vec;
    vec.resize(100);

    for(size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = static_cast<float>(i) * 0.5f;
    }

    for(size_t i = 0; i < vec.size(); ++i)
    {
        EXPECT_EQ(vec[i], static_cast<float>(i) * 0.5f);
    }
}

TEST(Allocators, PinnedHostAllocatorBasicOperations)
{
    SKIP_IF_NO_DEVICES();

    Pinned_host_allocator<int> alloc;

    // Allocate pinned memory for 10 integers
    auto ptr = alloc.allocate(10);
    ASSERT_NE(ptr, nullptr);

    // Construct objects
    for(int i = 0; i < 10; ++i)
    {
        alloc.construct(&ptr[i], i * 3);
    }

    // Verify values
    for(int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(ptr[i], i * 3);
    }

    // Destroy objects
    for(int i = 0; i < 10; ++i)
    {
        alloc.destroy(&ptr[i]);
    }

    // Deallocate memory
    alloc.deallocate(ptr, 10);
}

TEST(Allocators, PinnedHostAllocatorRebind)
{
    SKIP_IF_NO_DEVICES();

    Pinned_host_allocator<int> int_alloc;
    typename Pinned_host_allocator<int>::template rebind<double>::other double_alloc;

    auto int_ptr = int_alloc.allocate(5);
    auto double_ptr = double_alloc.allocate(5);

    ASSERT_NE(int_ptr, nullptr);
    ASSERT_NE(double_ptr, nullptr);

    int_alloc.deallocate(int_ptr, 5);
    double_alloc.deallocate(double_ptr, 5);
}

TEST(Allocators, DeviceAllocatorBasicOperations)
{
    SKIP_IF_NO_DEVICES();

    Device_allocator<float> alloc;

    // Allocate device memory for 100 floats
    auto ptr = alloc.allocate(100);
    ASSERT_NE(ptr, nullptr);

    // Verify we can use the pointer with HIP operations
    std::vector<float> host_data(100);
    for(size_t i = 0; i < host_data.size(); ++i)
    {
        host_data[i] = static_cast<float>(i);
    }

    hipError_t err = hipMemcpy(ptr, host_data.data(), 100 * sizeof(float), hipMemcpyHostToDevice);
    EXPECT_EQ(err, hipSuccess);

    std::vector<float> result(100);
    err = hipMemcpy(result.data(), ptr, 100 * sizeof(float), hipMemcpyDeviceToHost);
    EXPECT_EQ(err, hipSuccess);

    for(size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_EQ(result[i], static_cast<float>(i));
    }

    // Deallocate memory
    alloc.deallocate(ptr, 100);
}

TEST(Allocators, DeviceAllocatorRebind)
{
    SKIP_IF_NO_DEVICES();

    Device_allocator<int> int_alloc;
    typename Device_allocator<int>::template rebind<double>::other double_alloc;

    auto int_ptr = int_alloc.allocate(5);
    auto double_ptr = double_alloc.allocate(5);

    ASSERT_NE(int_ptr, nullptr);
    ASSERT_NE(double_ptr, nullptr);

    int_alloc.deallocate(int_ptr, 5);
    double_alloc.deallocate(double_ptr, 5);
}

TEST(Allocators, DeviceAllocatorComparison)
{
    SKIP_IF_NO_DEVICES();

    Device_allocator<int> alloc1;
    Device_allocator<int> alloc2;
    Device_allocator<double> alloc3;

    EXPECT_TRUE(alloc1 == alloc2);
    EXPECT_FALSE(alloc1 != alloc2);
    EXPECT_TRUE(alloc1 == alloc3);
}

// Test allocator traits compatibility
TEST(Allocators, AllocatorTraitsCompatibility)
{
    using HostTraits = std::allocator_traits<Host_allocator<int>>;
    using PinnedTraits = std::allocator_traits<Pinned_host_allocator<int>>;
    using DeviceTraits = std::allocator_traits<Device_allocator<int>>;

    // Check that all required types are defined
    static_assert(std::is_same_v<HostTraits::value_type, int>);
    static_assert(std::is_same_v<HostTraits::pointer, int*>);
    static_assert(std::is_same_v<HostTraits::const_pointer, const int*>);
    static_assert(std::is_same_v<HostTraits::size_type, std::size_t>);
    static_assert(std::is_same_v<HostTraits::difference_type, std::ptrdiff_t>);

    static_assert(std::is_same_v<PinnedTraits::value_type, int>);
    static_assert(std::is_same_v<PinnedTraits::pointer, int*>);
    static_assert(std::is_same_v<PinnedTraits::const_pointer, const int*>);
    static_assert(std::is_same_v<PinnedTraits::size_type, std::size_t>);
    static_assert(std::is_same_v<PinnedTraits::difference_type, std::ptrdiff_t>);

    static_assert(std::is_same_v<DeviceTraits::value_type, int>);
    static_assert(std::is_same_v<DeviceTraits::pointer, int*>);
    static_assert(std::is_same_v<DeviceTraits::const_pointer, const int*>);
    static_assert(std::is_same_v<DeviceTraits::size_type, std::size_t>);
    static_assert(std::is_same_v<DeviceTraits::difference_type, std::ptrdiff_t>);
}

// Test exception handling
TEST(Allocators, AllocationFailure)
{
    Host_allocator<int> alloc;

    // Try to allocate an impossibly large amount of memory
    EXPECT_THROW(std::ignore = alloc.allocate(std::numeric_limits<std::size_t>::max()),
                 std::bad_alloc);
}

TEST(Allocators, PinnedAllocationFailure)
{
    SKIP_IF_NO_DEVICES();

    Pinned_host_allocator<int> alloc;

    // Try to allocate an impossibly large amount of memory
    EXPECT_THROW(std::ignore = alloc.allocate(std::numeric_limits<std::size_t>::max()),
                 std::bad_alloc);
}

TEST(Allocators, DeviceAllocationFailure)
{
    SKIP_IF_NO_DEVICES();

    Device_allocator<int> alloc;

    // Try to allocate an impossibly large amount of memory
    EXPECT_THROW(std::ignore = alloc.allocate(std::numeric_limits<std::size_t>::max()),
                 std::bad_alloc);
}
