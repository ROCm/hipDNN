// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/AdamTestStuff.hpp>
#include <hipdnn_sdk/test_utilities/BatchnormSignatureRegistrar.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

TEST(TestAdamTestSTuff, BatchnormFwdInferenceFloat)
{
    Tensor<float> inputTensor({1, 3, 224, 224});
    Tensor<float> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    BatchnormSignatureKey key{
        .inputDataType = DataType::DataType_FLOAT,
        .nodeAttributesType = NodeAttributes_BatchnormInferenceAttributes,
    };

    auto it = batchnormRegistry().find(key);
    if(it != batchnormRegistry().end())
    {
        std::any inputAny = std::ref(inputTensor);
        std::any scaleAny = std::ref(scaleTensor);
        std::any biasAny = std::ref(biasTensor);
        std::any meanAny = std::ref(meanTensor);
        std::any varianceAny = std::ref(varianceTensor);
        std::any outputAny = std::ref(outputTensor);

        it->second(inputAny, scaleAny, biasAny, meanAny, varianceAny, outputAny, 1e-5);
    }
    else
    {
        throw std::runtime_error("No batchnorm implementation registered for this signature.");
    }
}

TEST(TestAdamTestSTuff, BatchnormFwdInferenceHalf)
{
    Tensor<half> inputTensor({1, 3, 224, 224});
    Tensor<half> outputTensor({1, 3, 224, 224});
    Tensor<half> biasTensor({1, 3});
    Tensor<half> scaleTensor({1, 3});
    Tensor<half> meanTensor({1, 3});
    Tensor<half> varianceTensor({1, 3});

    BatchnormSignatureKey key{
        .inputDataType = DataType::DataType_HALF,
        .nodeAttributesType = NodeAttributes_BatchnormInferenceAttributes,
    };

    auto it = batchnormRegistry().find(key);
    if(it != batchnormRegistry().end())
    {
        std::any inputAny = std::ref(inputTensor);
        std::any scaleAny = std::ref(scaleTensor);
        std::any biasAny = std::ref(biasTensor);
        std::any meanAny = std::ref(meanTensor);
        std::any varianceAny = std::ref(varianceTensor);
        std::any outputAny = std::ref(outputTensor);

        it->second(inputAny, scaleAny, biasAny, meanAny, varianceAny, outputAny, 1e-5);
    }
    else
    {
        throw std::runtime_error("No batchnorm implementation registered for this signature.");
    }
}
