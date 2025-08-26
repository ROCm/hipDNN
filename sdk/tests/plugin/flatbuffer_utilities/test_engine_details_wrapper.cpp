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
    EngineDetailsWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), Hipdnn_plugin_exception);
    EXPECT_THROW(wrapper.getEngineDetails(), Hipdnn_plugin_exception);
}

TEST(EngineDetailsWrapperTest, ValidBufferIsValid)
{
    int64_t test_engine_id = 42;
    auto builder = build_valid_engine_details_buffer(test_engine_id);
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.engineId(), test_engine_id);
    EXPECT_NO_THROW(wrapper.getEngineDetails());
}

TEST(EngineDetailsWrapperTest, CorruptedBufferIsNotValid)
{
    std::vector<uint8_t> buffer(16, 0xFF); // Not a valid flatbuffer
    EngineDetailsWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), Hipdnn_plugin_exception);
}
