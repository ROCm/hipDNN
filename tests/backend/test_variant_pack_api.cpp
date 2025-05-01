// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <cstring>
#include <gtest/gtest.h>

class Variant_pack_descriptor_api_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _varpack;

    void SetUp() override
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_varpack),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_varpack, nullptr);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_varpack), HIPDNN_STATUS_SUCCESS);
    }
};

TEST_F(Variant_pack_descriptor_api_tests, ValidSetAttributesAndFinalize)
{
    std::array<void*, 3> dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        dev_ptrs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _varpack, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, 3, uids.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendFinalize(_varpack), HIPDNN_STATUS_SUCCESS);
}

TEST_F(Variant_pack_descriptor_api_tests, ValidSetAttributesAndGetAttributesBeforeFinalize)
{
    std::array<void*, 3> dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        dev_ptrs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _varpack, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, 3, uids.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace),
        HIPDNN_STATUS_SUCCESS);

    std::array<void*, 3> retrieved_dev_ptrs;
    int64_t element_count = 0;

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_INT64,
                                        3,
                                        &element_count,
                                        retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(Variant_pack_descriptor_api_tests, InvalidSetAttributes)
{
    std::array<void*, 3> dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS expects HIPDNN_TYPE_VOID_PTR
    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_INT64,
                                        3,
                                        dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS expects HIPDNN_TYPE_INT64
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_VOID_PTR, 3, uids.data()),
        HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_WORKSPACE expects HIPDNN_TYPE_VOID_PTR
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 2, &workspace),
        HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since null ptr passed
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS, HIPDNN_TYPE_VOID_PTR, 3, nullptr),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Variant_pack_descriptor_api_tests, InvalidGetAttributes)
{
    std::array<void*, 3> retrieved_dev_ptrs;
    int64_t element_count = 0;

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_INT64,
                                        3,
                                        &element_count,
                                        retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

// Invalid finalize since no dev_ptrs, or ids were set
TEST_F(Variant_pack_descriptor_api_tests, InvalidFinalizeEmpty)
{
    EXPECT_EQ(hipdnnBackendFinalize(_varpack), HIPDNN_STATUS_BAD_PARAM);
}

// Invalid finalize since dev_ptrs, and ids do not have the same size.
TEST_F(Variant_pack_descriptor_api_tests, InvalidFinalizeParams)
{
    std::array<void*, 3> dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 2> uids = {1, 2};

    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        dev_ptrs.size(),
                                        dev_ptrs.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                        HIPDNN_TYPE_INT64,
                                        uids.size(),
                                        uids.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_varpack), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Variant_pack_descriptor_api_tests, InvalidNullDescriptor)
{
    hipdnnBackendDescriptor_t null_descriptor = nullptr;
    int64_t dummy_data = 42;
    hipdnnStatus_t status = hipdnnBackendSetAttribute(
        null_descriptor, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, 1, &dummy_data);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    int64_t retrieved_data = 0;
    int64_t element_count = 0;
    status = hipdnnBackendGetAttribute(null_descriptor,
                                       HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &element_count,
                                       &retrieved_data);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

class Finalized_variant_pack_descriptor_api_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _varpack;
    std::array<void*, 3> _dev_ptrs = {reinterpret_cast<void*>(0x1234),
                                      reinterpret_cast<void*>(0x5678),
                                      reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> _uids = {1, 2, 3};
    void* _workspace = reinterpret_cast<void*>(0xdeadbeef);

    void SetUp() override
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_varpack),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_varpack, nullptr);

        EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                            HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                            HIPDNN_TYPE_VOID_PTR,
                                            static_cast<int64_t>(_dev_ptrs.size()),
                                            _dev_ptrs.data()),
                  HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                            HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                            HIPDNN_TYPE_INT64,
                                            static_cast<int64_t>(_uids.size()),
                                            _uids.data()),
                  HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            hipdnnBackendSetAttribute(
                _varpack, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &_workspace),
            HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendFinalize(_varpack), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_varpack), HIPDNN_STATUS_SUCCESS);
    }
};

TEST_F(Finalized_variant_pack_descriptor_api_tests, ValidGetAttributes)
{
    std::array<void*, 3> retrieved_dev_ptrs;
    std::array<int64_t, 3> retrieved_uids;
    void* retrieved_workspace = nullptr;
    int64_t element_count = 0;

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        &element_count,
                                        retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(element_count, 3);
    EXPECT_EQ(
        std::memcmp(retrieved_dev_ptrs.data(), _dev_ptrs.data(), _dev_ptrs.size() * sizeof(void*)),
        0);

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                        HIPDNN_TYPE_INT64,
                                        3,
                                        &element_count,
                                        retrieved_uids.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(element_count, 3);
    EXPECT_EQ(std::memcmp(retrieved_uids.data(), _uids.data(), _uids.size() * sizeof(int64_t)), 0);

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                        HIPDNN_TYPE_VOID_PTR,
                                        1,
                                        &element_count,
                                        &retrieved_workspace),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(element_count, 1);
    EXPECT_EQ(retrieved_workspace, _workspace);
}

TEST_F(Finalized_variant_pack_descriptor_api_tests, InvalidGetAttributes)
{
    std::array<void*, 3> retrieved_dev_ptrs;
    std::array<int64_t, 3> retrieved_uids;
    void* retrieved_workspace = nullptr;
    int64_t element_count = 0;

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS expects HIPDNN_TYPE_VOID_PTR
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_INT64,
                                        3,
                                        &element_count,
                                        retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS expects HIPDNN_TYPE_INT64
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        &element_count,
                                        retrieved_uids.data()),
              HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_WORKSPACE expects HIPDNN_TYPE_VOID_PTR
    // but element_count should be 1
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                        HIPDNN_TYPE_VOID_PTR,
                                        2,
                                        &element_count,
                                        &retrieved_workspace),
              HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since array_of_elements cannot be nullptr
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        &element_count,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since element_count cannot be nullptr
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        nullptr,
                                        retrieved_dev_ptrs.data()),
              HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Finalized_variant_pack_descriptor_api_tests, InvalidSetAttributes)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        static_cast<int64_t>(_dev_ptrs.size()),
                                        _dev_ptrs.data()),
              HIPDNN_STATUS_NOT_INITIALIZED);
}
