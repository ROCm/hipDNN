// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

using namespace hipdnn_sdk::utilities;

TEST(TestShapeUtils, GenerateStridesNhwcValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> strideOrder = {3, 0, 2, 1}; // NHWC
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 1, 8, 2}));
}

TEST(TestShapeUtils, GenerateStridesNdhwcValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    std::vector<int64_t> strideOrder = {4, 0, 3, 2, 1}; // NDHWC
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 1, 40, 10, 2}));
}

TEST(TestShapeUtils, GenerateStridesNchwValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> strideOrder = {3, 2, 1, 0}; // NCHW
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 12, 4, 1}));
}

TEST(TestShapeUtils, GenerateStridesNcdhwValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    std::vector<int64_t> strideOrder = {4, 3, 2, 1, 0}; // NCDHW
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 60, 20, 5, 1}));
}

TEST(TestShapeUtils, GenerateStridesSingleDimension)
{
    std::vector<int64_t> dim = {5};
    std::vector<int64_t> strideOrder = {0};
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{1}));
}

TEST(TestShapeUtils, StrideOrderNhwcFiveDimensions)
{
    size_t numDims = 5;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{4, 0, 3, 2, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcFourDimensions)
{
    size_t numDims = 4;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{3, 0, 2, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcThreeDimensions)
{
    size_t numDims = 3;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{2, 0, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcWithTwoDimensions)
{
    size_t numDims = 2;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{1, 0}));
}

TEST(TestShapeUtils, StrideOrderNhwcWithSingleDimension)
{
    size_t numDims = 1;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{0}));
}

TEST(TestShapeUtils, GenerateStridesEmptyDimensions)
{
    std::vector<int64_t> dim = {};
    std::vector<int64_t> strideOrder = {};
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}

TEST(TestShapeUtils, BroadcastCompatibleExactMatch)
{
    std::vector<int64_t> inputDims = {2, 3, 4};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleInputNon1OutputIs1Invalid)
{
    // Input has non-1 value where output has 1 - invalid
    std::vector<int64_t> inputDims = {2, 3, 4};
    std::vector<int64_t> outputDims = {2, 1, 4};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleInputIs1OutputNon1Valid)
{
    // Input has 1 where output has non-1 - valid (broadcast)
    std::vector<int64_t> inputDims = {2, 1, 4};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleFewerInputDims)
{
    // Input has fewer dimensions - valid (implicit leading 1s)
    std::vector<int64_t> inputDims = {3, 4};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleMoreInputDimsInvalid)
{
    // Input has more dimensions than output - invalid
    std::vector<int64_t> inputDims = {2, 3, 4, 5};
    std::vector<int64_t> outputDims = {3, 4, 5};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleMismatchedDimsInvalid)
{
    // Input and output don't match (3 vs 5) - invalid
    std::vector<int64_t> inputDims = {2, 3, 4};
    std::vector<int64_t> outputDims = {2, 5, 4};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleEmptyInputValid)
{
    // Empty input is broadcastable to any output
    std::vector<int64_t> inputDims = {};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleEmptyOutputWithNonEmptyInputInvalid)
{
    // Non-empty input cannot broadcast to empty output
    std::vector<int64_t> inputDims = {2, 3};
    std::vector<int64_t> outputDims = {};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleBothEmpty)
{
    // Both empty is valid (exact match)
    std::vector<int64_t> inputDims = {};
    std::vector<int64_t> outputDims = {};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleScalarToTensor)
{
    // Scalar [1] broadcasts to any shape
    std::vector<int64_t> inputDims = {1};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleVectorToMatrix)
{
    // Vector [4] broadcasts to matrix [3, 4]
    std::vector<int64_t> inputDims = {4};
    std::vector<int64_t> outputDims = {3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleVectorMismatchInvalid)
{
    // Vector [5] cannot broadcast to matrix [3, 4]
    std::vector<int64_t> inputDims = {5};
    std::vector<int64_t> outputDims = {3, 4};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleComplexBroadcast)
{
    // Complex broadcast: [1, 3, 1] -> [2, 3, 4]
    std::vector<int64_t> inputDims = {1, 3, 1};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatiblePartialMatchWithOnes)
{
    // [2, 1, 4] -> [2, 3, 4]
    std::vector<int64_t> inputDims = {2, 1, 4};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleHigherDimensionalBroadcast)
{
    // 5D broadcast: [1, 1, 3, 1, 5] -> [2, 4, 3, 6, 5]
    std::vector<int64_t> inputDims = {1, 1, 3, 1, 5};
    std::vector<int64_t> outputDims = {2, 4, 3, 6, 5};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleLeadingOnesImplicit)
{
    // [3, 4] broadcasts to [1, 2, 3, 4] (implicit leading 1s)
    std::vector<int64_t> inputDims = {3, 4};
    std::vector<int64_t> outputDims = {1, 2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleSingleDimToMultiDim)
{
    // Single dimension [5] to multi-dimensional [2, 3, 5]
    std::vector<int64_t> inputDims = {5};
    std::vector<int64_t> outputDims = {2, 3, 5};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleAllOnesInput)
{
    // All 1s input broadcasts to any matching rank output
    std::vector<int64_t> inputDims = {1, 1, 1};
    std::vector<int64_t> outputDims = {5, 7, 9};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleMixedOnesAndMatches)
{
    // Mix of exact matches and 1s: [2, 1, 4, 1] -> [2, 3, 4, 5]
    std::vector<int64_t> inputDims = {2, 1, 4, 1};
    std::vector<int64_t> outputDims = {2, 3, 4, 5};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, GenerateStridesMoreDimsThanStridesThrows)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> strideOrder = {2, 1, 0};

    EXPECT_THROW(generateStrides(dim, strideOrder), std::invalid_argument);
}

TEST(TestShapeUtils, GenerateStridesLessDimsThanStridesValid)
{
    std::vector<int64_t> dim = {1, 2, 3};
    std::vector<int64_t> strideOrder = {3, 2, 1, 0};

    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{6, 3, 1}));
}

TEST(TestShapeUtils, GenerateStridesFewerDimsNhwcOrdering)
{
    std::vector<int64_t> dim = {2, 3, 4};
    std::vector<int64_t> strideOrder = {2, 0, 1, 3};

    // Should use first 3 elements: {2, 0, 1} for NHW ordering
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{12, 1, 3}));
}

TEST(TestShapeUtils, GenerateStridesEmptyDimsWithNonEmptyStridesValid)
{
    std::vector<int64_t> dim = {};
    std::vector<int64_t> strideOrder = {3, 2, 1, 0};

    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}
