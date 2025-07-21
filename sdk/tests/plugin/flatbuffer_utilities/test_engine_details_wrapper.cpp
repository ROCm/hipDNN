// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/engine_details_wrapper.hpp>

using namespace hipdnn_plugin;

flatbuffers::FlatBufferBuilder build_valid_engine_details_buffer(int64_t engine_id)
{
    flatbuffers::FlatBufferBuilder builder;
    auto config = hipdnn_sdk::data_objects::CreateEngineDetails(builder, engine_id);
    builder.Finish(config);
    return builder;
}

TEST(EngineDetailsWrapperTest, InvalidBufferIsNotValid)
{
    Engine_details_wrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.is_valid());
    EXPECT_THROW(wrapper.engine_id(), Hipdnn_plugin_exception);
    EXPECT_THROW(wrapper.get_engine_details(), Hipdnn_plugin_exception);
}

TEST(EngineDetailsWrapperTest, ValidBufferIsValid)
{
    int64_t test_engine_id = 42;
    auto builder = build_valid_engine_details_buffer(test_engine_id);
    Engine_details_wrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.is_valid());
    EXPECT_EQ(wrapper.engine_id(), test_engine_id);
    EXPECT_NO_THROW(wrapper.get_engine_details());
}

TEST(EngineDetailsWrapperTest, CorruptedBufferIsNotValid)
{
    std::vector<uint8_t> buffer(16, 0xFF); // Not a valid flatbuffer
    Engine_details_wrapper wrapper(buffer.data(), buffer.size());
    EXPECT_FALSE(wrapper.is_valid());
    EXPECT_THROW(wrapper.engine_id(), Hipdnn_plugin_exception);
}
