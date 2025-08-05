// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <memory>

#include "hipdnn_engine_plugin_handle.hpp"

class Hipdnn_engine_plugin_handle_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _handle = std::make_unique<hipdnnEnginePluginHandle>();
    }

    std::unique_ptr<hipdnnEnginePluginHandle> _handle;
};

TEST_F(Hipdnn_engine_plugin_handle_test, DefaultConstruction)
{
    EXPECT_EQ(_handle->miopen_handle, nullptr);
    EXPECT_EQ(_handle->miopen_container, nullptr);
    EXPECT_EQ(_handle->get_stream(), nullptr);
}

TEST_F(Hipdnn_engine_plugin_handle_test, StoreDetachedBuffer)
{
    flatbuffers::FlatBufferBuilder builder;
    auto created_string = builder.CreateString("test");
    builder.Finish(created_string);
    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    const void* ptr = reinterpret_cast<const void*>(0x12354);

    _handle->store_engine_details_detached_buffer(ptr, std::move(buffer));

    // Buffer should be stored, verify by trying to remove it
    _handle->remove_engine_details_detached_buffer(ptr);
}

TEST_F(Hipdnn_engine_plugin_handle_test, RemoveDetachedBuffer)
{
    flatbuffers::FlatBufferBuilder builder;
    auto created_string = builder.CreateString("test");
    builder.Finish(created_string);
    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    const void* ptr = reinterpret_cast<const void*>(0x12354);

    _handle->store_engine_details_detached_buffer(ptr, std::move(buffer));
    _handle->remove_engine_details_detached_buffer(ptr);

    // Should not crash when removing non-existent buffer
    _handle->remove_engine_details_detached_buffer(ptr);
}

TEST_F(Hipdnn_engine_plugin_handle_test, RemoveNonExistentBuffer)
{
    const void* fake_ptr = reinterpret_cast<const void*>(0x12345678);

    // Should not crash when removing non-existent buffer
    EXPECT_NO_THROW(_handle->remove_engine_details_detached_buffer(fake_ptr));
}

TEST_F(Hipdnn_engine_plugin_handle_test, MultipleBuffers)
{
    flatbuffers::FlatBufferBuilder builder1;
    auto created_string1 = builder1.CreateString("test");
    builder1.Finish(created_string1);
    auto buffer1 = std::make_unique<flatbuffers::DetachedBuffer>(builder1.Release());
    const void* ptr1 = reinterpret_cast<const void*>(0x12354);

    flatbuffers::FlatBufferBuilder builder2;
    auto created_string2 = builder2.CreateString("test2");
    builder2.Finish(created_string2);
    auto buffer2 = std::make_unique<flatbuffers::DetachedBuffer>(builder2.Release());
    const void* ptr2 = reinterpret_cast<const void*>(0x54311);

    _handle->store_engine_details_detached_buffer(ptr1, std::move(buffer1));
    _handle->store_engine_details_detached_buffer(ptr2, std::move(buffer2));

    _handle->remove_engine_details_detached_buffer(ptr1);
    _handle->remove_engine_details_detached_buffer(ptr2);
}

TEST_F(Hipdnn_engine_plugin_handle_test, ThrowsWithNoMiopenHandle)
{
    EXPECT_THROW(_handle->set_stream(nullptr), hipdnn_plugin::Hipdnn_plugin_exception);
}
