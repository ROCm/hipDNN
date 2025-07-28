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
    auto mock_ptr = std::make_shared<Mock_descriptor<Engine_descriptor>>();

    Scoped_descriptor packed(hipdnnBackendDescriptor::pack_descriptor(mock_ptr));
    ASSERT_NE(packed.get(), nullptr);

    auto unpacked = hipdnnBackendDescriptor::unpack_descriptor<Mock_descriptor<Engine_descriptor>>(
        packed.get(), HIPDNN_STATUS_INTERNAL_ERROR, "fail");
    ASSERT_EQ(unpacked.get(), mock_ptr.get());
    ASSERT_EQ(unpacked->get_type(), HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
}

TEST(BackendDescriptorTest, AsDescriptorCastsCorrectly)
{
    auto mock_ptr = std::make_shared<Mock_descriptor<Engine_descriptor>>();
    Scoped_descriptor packed(hipdnnBackendDescriptor::pack_descriptor(mock_ptr));

    auto result = packed.get()->as_descriptor<Mock_descriptor<Engine_descriptor>>();
    ASSERT_EQ(result.get(), mock_ptr.get());
}

TEST(BackendDescriptorTest, UnpackDescriptorFromArrayWorks)
{
    auto mock_ptr = std::make_shared<Mock_descriptor<Engine_descriptor>>();
    Scoped_descriptor packed(hipdnnBackendDescriptor::pack_descriptor(mock_ptr));

    void* array_of_elements = &packed.descriptor;
    auto unpacked = hipdnnBackendDescriptor::unpack_descriptor<Mock_descriptor<Engine_descriptor>>(
        array_of_elements, HIPDNN_STATUS_INTERNAL_ERROR, "fail");
    ASSERT_EQ(unpacked.get(), mock_ptr.get());
}

TEST(BackendDescriptorTest, PackDescriptorToArrayWorks)
{
    auto mock_ptr = std::make_shared<Mock_descriptor<Engine_descriptor>>();
    hipdnnBackendDescriptor_t desc = nullptr;
    void* array_of_elements = &desc;
    hipdnnBackendDescriptor::pack_descriptor(mock_ptr, array_of_elements);
    Scoped_descriptor scoped(desc);

    ASSERT_NE(desc, nullptr);
}

TEST(BackendDescriptorTest, UnpackDescriptorThrowsOnNullDescriptor)
{
    EXPECT_THROW(
        {
            hipdnnBackendDescriptor::unpack_descriptor<Mock_descriptor<Engine_descriptor>>(
                static_cast<hipdnnBackendDescriptor*>(nullptr),
                HIPDNN_STATUS_INTERNAL_ERROR,
                "fail");
        },
        Hipdnn_exception);
}

TEST(BackendDescriptorTest, UnpackDescriptorThrowsOnNullPrivateDescriptor)
{
    Scoped_descriptor packed(new hipdnnBackendDescriptor());

    EXPECT_THROW(
        {
            hipdnnBackendDescriptor::unpack_descriptor<Mock_descriptor<Engine_descriptor>>(
                packed.get(), HIPDNN_STATUS_INTERNAL_ERROR, "fail");
        },
        Hipdnn_exception);
}

TEST(BackendDescriptorTest, UnpackDescriptorFromArrayThrowsOnNullArray)
{
    EXPECT_THROW(
        {
            hipdnnBackendDescriptor::unpack_descriptor<Mock_descriptor<Engine_descriptor>>(
                static_cast<void*>(nullptr), HIPDNN_STATUS_INTERNAL_ERROR, "fail");
        },
        Hipdnn_exception);
}
