// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/backend/scoped_hipdnn_backend_descriptor.hpp>
#include <hipdnn_sdk/utilities/string_util.hpp>

#include "fake_backend/mock_hipdnn_backend.hpp"

using namespace hipdnn_frontend;
using namespace ::testing;

class Hipdnn_backend_descriptor_test_fixture : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mock_backend;

    void SetUp() override
    {
        _mock_backend = std::make_shared<Mock_hipdnn_backend>();
        Hipdnn_backend_interface::set_instance(_mock_backend);

        ON_CALL(*_mock_backend, get_last_error_string(_, _))
            .WillByDefault([](char* error_string, size_t size) {
                std::string fake_error = "Fake backend error";
                hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
                    error_string, fake_error.c_str(), size - 1);
            });
    }
    void TearDown() override
    {
        Hipdnn_backend_interface::reset_instance();
        _mock_backend.reset();
    }
};

TEST_F(Hipdnn_backend_descriptor_test_fixture, DefaultConstructorIsInvalid)
{
    auto desc = Scoped_hipdnn_backend_descriptor();
    EXPECT_FALSE(desc.valid());
    EXPECT_EQ(desc.get(), nullptr);
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, ConstructWithRawDescriptorIsValid)
{
    auto fake_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    auto desc = Scoped_hipdnn_backend_descriptor(fake_desc);

    EXPECT_CALL(*_mock_backend, backend_destroy_descriptor(fake_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_TRUE(desc.valid());
    EXPECT_EQ(desc.get(), fake_desc);
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, ConstructWithTypeSuccess)
{
    auto fake_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mock_backend, backend_create_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fake_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fake_desc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend, backend_destroy_descriptor(fake_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc = Scoped_hipdnn_backend_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    EXPECT_TRUE(desc.valid());
    EXPECT_EQ(desc.get(), fake_desc);
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, ConstructWithTypeFailure)
{
    EXPECT_CALL(*_mock_backend, backend_create_descriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce([](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t*) {
            return HIPDNN_STATUS_BAD_PARAM;
        });
    EXPECT_CALL(*_mock_backend, get_last_error_string).Times(AnyNumber()); // Uninteresting call

    auto desc = Scoped_hipdnn_backend_descriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    EXPECT_FALSE(desc.valid());
    EXPECT_EQ(desc.get(), nullptr);
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, ConstructWithSerializedGraphSuccess)
{
    auto fake_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4321);
    std::vector<uint8_t> fake_graph{1, 2, 3, 4};
    EXPECT_CALL(*_mock_backend, backend_create_and_deserialize_graph_ext(_, _, _))
        .WillOnce([&fake_desc](hipdnnBackendDescriptor_t* out, const uint8_t*, size_t) {
            *out = fake_desc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend, backend_destroy_descriptor(fake_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc = Scoped_hipdnn_backend_descriptor(fake_graph.data(), fake_graph.size());
    EXPECT_TRUE(desc.valid());
    EXPECT_EQ(desc.get(), fake_desc);
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, ConstructWithSerializedGraphFailure)
{
    std::vector<uint8_t> fake_graph{1, 2, 3, 4};
    EXPECT_CALL(*_mock_backend, backend_create_and_deserialize_graph_ext(_, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t*, const uint8_t*, size_t) {
            return HIPDNN_STATUS_BAD_PARAM;
        });
    EXPECT_CALL(*_mock_backend, get_last_error_string).Times(AnyNumber()); // Uninteresting call

    auto desc = Scoped_hipdnn_backend_descriptor(fake_graph.data(), fake_graph.size());
    EXPECT_FALSE(desc.valid());
    EXPECT_EQ(desc.get(), nullptr);
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, DestructorDestroysDescriptorIfValid)
{
    auto fake_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9999);
    EXPECT_CALL(*_mock_backend, backend_create_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fake_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fake_desc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend, backend_destroy_descriptor(fake_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    {
        auto desc = Scoped_hipdnn_backend_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        EXPECT_TRUE(desc.valid());
    }
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, DestructorHandlesDestroyFailure)
{
    auto fake_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x8888);
    EXPECT_CALL(*_mock_backend, backend_create_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fake_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fake_desc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend, backend_destroy_descriptor(fake_desc))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));
    EXPECT_CALL(*_mock_backend, get_last_error_string).Times(AnyNumber()); // Uninteresting call

    {
        auto desc = Scoped_hipdnn_backend_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        EXPECT_TRUE(desc.valid());
    }
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, MoveConstructorTransfersOwnership)
{
    auto fake_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1111);
    EXPECT_CALL(*_mock_backend, backend_create_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fake_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fake_desc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend, backend_destroy_descriptor(fake_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc1 = Scoped_hipdnn_backend_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    EXPECT_TRUE(desc1.valid());
    auto raw = desc1.get();

    auto desc2 = Scoped_hipdnn_backend_descriptor(std::move(desc1));
    EXPECT_TRUE(desc2.valid());
    EXPECT_EQ(desc2.get(), raw);
    EXPECT_FALSE(desc1.valid());
    EXPECT_EQ(desc1.get(), nullptr);
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, MoveAssignmentTransfersOwnership)
{
    auto fake_desc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2222);
    auto fake_desc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3333);

    EXPECT_CALL(*_mock_backend, backend_create_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fake_desc1](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fake_desc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&fake_desc2](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fake_desc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mock_backend, backend_destroy_descriptor(fake_desc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend, backend_destroy_descriptor(fake_desc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc1 = Scoped_hipdnn_backend_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    auto desc2 = Scoped_hipdnn_backend_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);

    desc2 = std::move(desc1);

    EXPECT_TRUE(desc2.valid());
    EXPECT_EQ(desc2.get(), fake_desc1);
    EXPECT_FALSE(desc1.valid());
    EXPECT_EQ(desc1.get(), nullptr);
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, MoveFromInvalidDescriptor)
{
    auto desc1 = Scoped_hipdnn_backend_descriptor();
    auto desc2 = Scoped_hipdnn_backend_descriptor(std::move(desc1));
    EXPECT_FALSE(desc2.valid());
    EXPECT_EQ(desc2.get(), nullptr);
}

TEST_F(Hipdnn_backend_descriptor_test_fixture, MoveAssignFromInvalidDescriptor)
{
    auto fake_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5555);

    EXPECT_CALL(*_mock_backend, backend_create_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fake_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fake_desc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mock_backend, backend_destroy_descriptor(fake_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc1 = Scoped_hipdnn_backend_descriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    auto desc2 = Scoped_hipdnn_backend_descriptor();

    desc1 = std::move(desc2);

    EXPECT_FALSE(desc1.valid());
    EXPECT_EQ(desc1.get(), nullptr);
}
