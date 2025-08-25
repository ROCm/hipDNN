// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/backend_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/scoped_descriptor.hpp"
#include "mocks/mock_descriptor.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace ::testing;
using namespace hipdnn_backend;

TEST(BackendDescriptorTest, PackAndUnpackDescriptorWorks)
{
    auto mockPtr = std::make_shared<MockDescriptor<EngineDescriptor>>();

    ScopedDescriptor packed(HipdnnBackendDescriptor::packDescriptor(mockPtr));
    ASSERT_NE(packed.get(), nullptr);

    auto unpacked = HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
        packed.get(), HIPDNN_STATUS_INTERNAL_ERROR, "fail");
    ASSERT_EQ(unpacked.get(), mockPtr.get());
    ASSERT_EQ(unpacked->getType(), HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
}

TEST(BackendDescriptorTest, AsDescriptorCastsCorrectly)
{
    auto mockPtr = std::make_shared<MockDescriptor<EngineDescriptor>>();
    ScopedDescriptor packed(HipdnnBackendDescriptor::packDescriptor(mockPtr));

    auto result = packed.get()->asDescriptor<MockDescriptor<EngineDescriptor>>();
    ASSERT_EQ(result.get(), mockPtr.get());
}

TEST(BackendDescriptorTest, UnpackDescriptorFromArrayWorks)
{
    auto mockPtr = std::make_shared<MockDescriptor<EngineDescriptor>>();
    ScopedDescriptor packed(HipdnnBackendDescriptor::packDescriptor(mockPtr));

    void* arrayOfElements = &packed.descriptor;
    auto unpacked = HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
        arrayOfElements, HIPDNN_STATUS_INTERNAL_ERROR, "fail");
    ASSERT_EQ(unpacked.get(), mockPtr.get());
}

TEST(BackendDescriptorTest, PackDescriptorToArrayWorks)
{
    auto mockPtr = std::make_shared<MockDescriptor<EngineDescriptor>>();
    hipdnnBackendDescriptor_t desc = nullptr;
    void* arrayOfElements = &desc;
    HipdnnBackendDescriptor::packDescriptor(mockPtr, arrayOfElements);
    ScopedDescriptor scoped(desc);

    ASSERT_NE(desc, nullptr);
}

TEST(BackendDescriptorTest, UnpackDescriptorThrowsOnNullDescriptor)
{
    EXPECT_THROW(
        {
            HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
                static_cast<HipdnnBackendDescriptor*>(nullptr),
                HIPDNN_STATUS_INTERNAL_ERROR,
                "fail");
        },
        Hipdnn_exception);
}

TEST(BackendDescriptorTest, UnpackDescriptorThrowsOnNullPrivateDescriptor)
{
    ScopedDescriptor packed(new HipdnnBackendDescriptor());

    EXPECT_THROW(
        {
            HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
                packed.get(), HIPDNN_STATUS_INTERNAL_ERROR, "fail");
        },
        Hipdnn_exception);
}

TEST(BackendDescriptorTest, UnpackDescriptorFromArrayThrowsOnNullArray)
{
    EXPECT_THROW(
        {
            HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
                static_cast<void*>(nullptr), HIPDNN_STATUS_INTERNAL_ERROR, "fail");
        },
        Hipdnn_exception);
}
