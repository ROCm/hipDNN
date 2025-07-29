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
    EXPECT_EQ(_handle->stream, nullptr);
    EXPECT_EQ(_handle->miopen_container, nullptr);
}

TEST_F(Hipdnn_engine_plugin_handle_test, StoreDetachedBuffer)
{
    auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
    builder->CreateString("test");
    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder->Release());
    const void* ptr = buffer->data();

    _handle->store_detached_buffer(ptr, std::move(buffer));

    // Buffer should be stored, verify by trying to remove it
    _handle->remove_detached_buffer(ptr);
}

TEST_F(Hipdnn_engine_plugin_handle_test, RemoveDetachedBuffer)
{
    auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
    builder->CreateString("test");
    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder->Release());
    const void* ptr = buffer->data();

    _handle->store_detached_buffer(ptr, std::move(buffer));
    _handle->remove_detached_buffer(ptr);

    // Should not crash when removing non-existent buffer
    _handle->remove_detached_buffer(ptr);
}

TEST_F(Hipdnn_engine_plugin_handle_test, RemoveNonExistentBuffer)
{
    const void* fake_ptr = reinterpret_cast<const void*>(0x12345678);

    // Should not crash when removing non-existent buffer
    EXPECT_NO_THROW(_handle->remove_detached_buffer(fake_ptr));
}

TEST_F(Hipdnn_engine_plugin_handle_test, MultipleBuffers)
{
    auto builder1 = std::make_unique<flatbuffers::FlatBufferBuilder>();
    builder1->CreateString("test1");
    auto buffer1 = std::make_unique<flatbuffers::DetachedBuffer>(builder1->Release());
    const void* ptr1 = buffer1->data();

    auto builder2 = std::make_unique<flatbuffers::FlatBufferBuilder>();
    builder2->CreateString("test2");
    auto buffer2 = std::make_unique<flatbuffers::DetachedBuffer>(builder2->Release());
    const void* ptr2 = buffer2->data();

    _handle->store_detached_buffer(ptr1, std::move(buffer1));
    _handle->store_detached_buffer(ptr2, std::move(buffer2));

    _handle->remove_detached_buffer(ptr1);
    _handle->remove_detached_buffer(ptr2);
}
