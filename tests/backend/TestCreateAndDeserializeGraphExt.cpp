#include "hipdnn_backend.h"
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

// TODO: Move this into backend/tests
TEST(TestCreateAndDeserializeGraphExt, NullptrGraph)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(&descriptor, nullptr, 0);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(descriptor, nullptr);
}
