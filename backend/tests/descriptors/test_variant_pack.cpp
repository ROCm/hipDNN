// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "descriptors/variant_descriptor.hpp"
#include <gtest/gtest.h>

namespace hipdnn_backend
{

class Initialize_variant_pack_descriptor_tests : public ::testing::Test
{
protected:
    Variant_descriptor descriptor;

    void SetUp() override
    {
        ASSERT_FALSE(descriptor.is_finalized());
    }
};

TEST_F(Initialize_variant_pack_descriptor_tests, ValidSetAttributes)
{
    std::array<void*, 3> dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    EXPECT_EQ(descriptor.set_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       dev_ptrs.size(),
                                       dev_ptrs.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(descriptor.set_attribute(
                  HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, uids.size(), uids.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(descriptor.set_attribute(
                  HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(descriptor.finalize(), HIPDNN_STATUS_SUCCESS);
    EXPECT_TRUE(descriptor.is_finalized());
}

TEST_F(Initialize_variant_pack_descriptor_tests, ValidSetAndGetBeforeFinalAttributes)
{
    std::array<void*, 3> dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    EXPECT_EQ(descriptor.set_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       dev_ptrs.size(),
                                       dev_ptrs.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(descriptor.set_attribute(
                  HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, uids.size(), uids.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(descriptor.set_attribute(
                  HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace),
              HIPDNN_STATUS_SUCCESS);
    // getting before finalized
    std::array<void*, 3> retrieved_dev_ptrs;
    int64_t element_count = 0;

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       retrieved_dev_ptrs.size(),
                                       &element_count,
                                       retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(Initialize_variant_pack_descriptor_tests, InvalidSetAttributes)
{
    std::array<void*, 3> dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    EXPECT_EQ(descriptor.set_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_INT64,
                                       dev_ptrs.size(),
                                       dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(
        descriptor.set_attribute(
            HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_VOID_PTR, uids.size(), uids.data()),
        HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(descriptor.set_attribute(
                  HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 2, &workspace),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(
        descriptor.set_attribute(
            HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS, HIPDNN_TYPE_VOID_PTR, dev_ptrs.size(), nullptr),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Initialize_variant_pack_descriptor_tests, InvalidFinalizeCounts)
{
    std::array<void*, 3> dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 2> uids = {1, 2};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    EXPECT_EQ(descriptor.set_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       dev_ptrs.size(),
                                       dev_ptrs.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(descriptor.set_attribute(
                  HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, uids.size(), uids.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(descriptor.set_attribute(
                  HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);
    EXPECT_FALSE(descriptor.is_finalized());
}

TEST_F(Initialize_variant_pack_descriptor_tests, InvalidFinalizeUnsetParams)
{
    EXPECT_EQ(descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);
    EXPECT_FALSE(descriptor.is_finalized());
}

TEST_F(Initialize_variant_pack_descriptor_tests, InvalidGetAttributeNotFinalized)
{
    std::array<void*, 3> retrieved_dev_ptrs;
    int64_t element_count = 0;

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       retrieved_dev_ptrs.size(),
                                       &element_count,
                                       retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

class Finalized_variant_pack_descriptor_tests : public ::testing::Test
{
protected:
    Variant_descriptor descriptor;
    std::array<void*, 3> _dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                      reinterpret_cast<void*>(0x5678),
                                      reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> _uids = {1, 2, 3};
    void* _workspace = reinterpret_cast<void*>(0xdeadbeef);

    void SetUp() override
    {
        EXPECT_EQ(descriptor.set_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                           HIPDNN_TYPE_VOID_PTR,
                                           static_cast<int64_t>(_dev_ptrs.size()),
                                           _dev_ptrs.data()),
                  HIPDNN_STATUS_SUCCESS);

        EXPECT_EQ(descriptor.set_attribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                           HIPDNN_TYPE_INT64,
                                           static_cast<int64_t>(_uids.size()),
                                           _uids.data()),
                  HIPDNN_STATUS_SUCCESS);

        EXPECT_EQ(descriptor.set_attribute(
                      HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &_workspace),
                  HIPDNN_STATUS_SUCCESS);

        EXPECT_EQ(descriptor.finalize(), HIPDNN_STATUS_SUCCESS);
        EXPECT_TRUE(descriptor.is_finalized());
    }
};

TEST_F(Finalized_variant_pack_descriptor_tests, InvalidSetAttribute)
{
    EXPECT_EQ(descriptor.set_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       static_cast<int64_t>(_dev_ptrs.size()),
                                       _dev_ptrs.data()),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Finalized_variant_pack_descriptor_tests, ValidGetAttributes)
{
    std::array<void*, 3> retrieved_dev_ptrs;
    std::array<int64_t, 3> retrieved_uids;
    void* retrieved_workspace = nullptr;
    int64_t element_count = 0;

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       retrieved_dev_ptrs.size(),
                                       &element_count,
                                       retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(element_count, _dev_ptrs.size());
    EXPECT_EQ(
        std::memcmp(retrieved_dev_ptrs.data(), _dev_ptrs.data(), _dev_ptrs.size() * sizeof(void*)),
        0);

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                       HIPDNN_TYPE_INT64,
                                       retrieved_uids.size(),
                                       &element_count,
                                       retrieved_uids.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(element_count, _uids.size());
    EXPECT_EQ(std::memcmp(retrieved_uids.data(), _uids.data(), _uids.size() * sizeof(int64_t)), 0);

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                       HIPDNN_TYPE_VOID_PTR,
                                       1,
                                       &element_count,
                                       &retrieved_workspace),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(element_count, 1);
    EXPECT_EQ(retrieved_workspace, _workspace);
}

TEST_F(Finalized_variant_pack_descriptor_tests, InvalidGetAttributes)
{
    std::array<void*, 3> retrieved_dev_ptrs;
    std::array<int64_t, 3> retrieved_uids;
    void* retrieved_workspace = nullptr;
    int64_t element_count = 0;

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_INT64,
                                       retrieved_dev_ptrs.size(),
                                       &element_count,
                                       retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       retrieved_uids.size(),
                                       &element_count,
                                       retrieved_uids.data()),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                       HIPDNN_TYPE_VOID_PTR,
                                       2,
                                       &element_count,
                                       &retrieved_workspace),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       retrieved_dev_ptrs.size(),
                                       &element_count,
                                       nullptr),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(descriptor.get_attribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       HIPDNN_TYPE_VOID_PTR,
                                       retrieved_dev_ptrs.size(),
                                       nullptr,
                                       retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM);
}

}
