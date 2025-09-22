// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <limits>
#include <vector>

#include <hipdnn_sdk/plugin/PluginException.hpp>

#include "MiopenConvDescriptor.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

namespace
{

void copyWithCheck(const flatbuffers::Vector<int64_t>* src,
                   std::vector<int>& dst,
                   size_t expectedSize,
                   const char* name,
                   const char* expectedSizeName)
{
    if(src != nullptr)
    {
        if(src->size() != expectedSize)
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MiopenConvDescriptor: " + std::string(name)
                                                           + " size must be equal to "
                                                           + std::string(expectedSizeName));
        }
        if(!std::ranges::all_of(*src,
                                [](int64_t v) { return v <= std::numeric_limits<int>::max(); }))
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MiopenConvDescriptor: " + std::string(name)
                                                           + " values must be less than INT_MAX");
        }
        std::ranges::copy(*src, dst.begin());
    }
}

}

MiopenConvDescriptor::MiopenConvDescriptor(
    size_t spatialDimCount, const hipdnn_sdk::data_objects::ConvolutionFwdAttributes& attributes)
{
    if(spatialDimCount > std::numeric_limits<int>::max())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenConvDescriptor: spatialDimCount must be not greater than INT_MAX");
    }

    const auto convMode = attributes.conv_mode();
    if(convMode != hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenConvDescriptor: only ConvMode::CROSS_CORRELATION is supported");
    }

    if((attributes.pre_padding() == nullptr) != (attributes.post_padding() == nullptr))
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenConvDescriptor: both pre_padding and post_padding should be set or both "
            "should be null");
    }

    if(attributes.pre_padding() != nullptr && attributes.post_padding() != nullptr)
    {
        if(attributes.pre_padding()->size() != attributes.post_padding()->size())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "MiopenConvDescriptor: pre_padding and post_padding sizes must be equal");
        }

        if(!std::ranges::equal(*attributes.pre_padding(), *attributes.post_padding()))
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "MiopenConvDescriptor: asymmetric padding is not supported");
        }
    }

    const auto attrPadding = attributes.pre_padding();
    const auto attrStride = attributes.stride();
    const auto attrDilation = attributes.dilation();

    std::vector<int> padding(spatialDimCount, 0);
    std::vector<int> stride(spatialDimCount, 1);
    std::vector<int> dilation(spatialDimCount, 1);

    copyWithCheck(attrPadding, padding, spatialDimCount, "attrPadding", "spatialDimCount");
    copyWithCheck(attrStride, stride, spatialDimCount, "attrStride", "spatialDimCount");
    copyWithCheck(attrDilation, dilation, spatialDimCount, "attrDilation", "spatialDimCount");

    THROW_ON_MIOPEN_FAILURE(miopenCreateConvolutionDescriptor(&_descriptor));
    THROW_ON_MIOPEN_FAILURE(miopenInitConvolutionNdDescriptor(_descriptor,
                                                              static_cast<int>(spatialDimCount),
                                                              padding.data(),
                                                              stride.data(),
                                                              dilation.data(),
                                                              miopenConvolution));
}

MiopenConvDescriptor::MiopenConvDescriptor(MiopenConvDescriptor&& other) noexcept
    : _descriptor(other._descriptor)
{
    other._descriptor = nullptr;
}

MiopenConvDescriptor& MiopenConvDescriptor::operator=(MiopenConvDescriptor&& other) noexcept
{
    if(this != &other)
    {
        if(_descriptor != nullptr)
        {
            LOG_ON_MIOPEN_FAILURE(miopenDestroyConvolutionDescriptor(_descriptor));
        }

        _descriptor = other._descriptor;
        other._descriptor = nullptr;
    }
    return *this;
}

MiopenConvDescriptor::~MiopenConvDescriptor()
{
    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyConvolutionDescriptor(_descriptor));
    }
}

miopenConvolutionDescriptor_t MiopenConvDescriptor::convDescriptor() const
{
    return _descriptor;
}

}
