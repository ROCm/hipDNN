// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <HipdnnBackendHeuristicType.h>
#include <hipdnn_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_frontend
{

enum class ConvolutionMode
{
    NOT_SET = 0,
    CROSS_CORRELATION = 1,
    CONVOLUTION = 2
};
typedef ConvolutionMode ConvolutionMode_t; // NOLINT(readability-identifier-naming)

enum class PointwiseMode
{
    NOT_SET = 0,
    ABS = 1,
    ADD = 2,
    ADD_SQUARE = 3,
    BINARY_SELECT = 4,
    CEIL = 5,
    CMP_EQ = 6,
    CMP_GE = 7,
    CMP_GT = 8,
    CMP_LE = 9,
    CMP_LT = 10,
    CMP_NEQ = 11,
    DIV = 12,
    ELU_BWD = 13,
    ELU_FWD = 14,
    ERF = 15,
    EXP = 16,
    FLOOR = 17,
    GELU_APPROX_TANH_BWD = 18,
    GELU_APPROX_TANH_FWD = 19,
    GELU_BWD = 20,
    GELU_FWD = 21,
    GEN_INDEX = 22,
    IDENTITY = 23,
    LOG = 24,
    LOGICAL_AND = 25,
    LOGICAL_NOT = 26,
    LOGICAL_OR = 27,
    MAX = 28,
    MIN = 29,
    MUL = 30,
    NEG = 31,
    RECIPROCAL = 32,
    RELU_BWD = 33,
    RELU_FWD = 34,
    RSQRT = 35,
    SIGMOID_BWD = 36,
    SIGMOID_FWD = 37,
    SIN = 38,
    SOFTPLUS_BWD = 39,
    SOFTPLUS_FWD = 40,
    SQRT = 41,
    SUB = 42,
    SWISH_BWD = 43,
    SWISH_FWD = 44,
    TAN = 45,
    TANH_BWD = 46,
    TANH_FWD = 47,
};
typedef PointwiseMode PointwiseMode_t; // NOLINT(readability-identifier-naming)

enum class DataType
{
    NOT_SET = 0,
    FLOAT = 1,
    HALF = 2,
    BFLOAT16 = 3,
    DOUBLE = 4,
    UINT8 = 5,
    INT32 = 6,
};
typedef DataType DataType_t; // NOLINT(readability-identifier-naming)

enum class HeuristicMode
{
    FALLBACK,
};
typedef HeuristicMode HeurMode_t; // NOLINT(readability-identifier-naming)

template <typename T>
DataType getDataTypeEnumFromType()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return DataType::HALF;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return DataType::BFLOAT16;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return DataType::DOUBLE;
    }
    else if constexpr(std::is_same_v<T, uint8_t>)
    {
        return DataType::UINT8;
    }
    else if constexpr(std::is_same_v<T, int32_t>)
    {
        return DataType::INT32;
    }
    else
    {
        return DataType::NOT_SET;
    }
}

inline hipdnn_sdk::data_objects::ConvMode toSdkType(const ConvolutionMode& type)
{
    switch(type)
    {
    case ConvolutionMode::CROSS_CORRELATION:
        return hipdnn_sdk::data_objects::ConvMode::ConvMode_CROSS_CORRELATION;
    case ConvolutionMode::CONVOLUTION:
        return hipdnn_sdk::data_objects::ConvMode::ConvMode_CONVOLUTION;
    default:
        return hipdnn_sdk::data_objects::ConvMode::ConvMode_UNSET;
    }
}

inline hipdnn_sdk::data_objects::DataType toSdkType(const DataType& type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return hipdnn_sdk::data_objects::DataType::DataType_FLOAT;
    case DataType::HALF:
        return hipdnn_sdk::data_objects::DataType::DataType_HALF;
    case DataType::BFLOAT16:
        return hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16;
    case DataType::DOUBLE:
        return hipdnn_sdk::data_objects::DataType::DataType_DOUBLE;
    case DataType::UINT8:
        return hipdnn_sdk::data_objects::DataType::DataType_UINT8;
    case DataType::INT32:
        return hipdnn_sdk::data_objects::DataType::DataType_INT32;
    default:
        return hipdnn_sdk::data_objects::DataType::DataType_UNSET;
    }
}

inline hipdnn_sdk::data_objects::PointwiseMode toSdkType(const PointwiseMode& type)
{
    switch(type)
    {
    case PointwiseMode::ABS:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_ABS;
    case PointwiseMode::ADD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_ADD;
    case PointwiseMode::ADD_SQUARE:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_ADD_SQUARE;
    case PointwiseMode::BINARY_SELECT:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_BINARY_SELECT;
    case PointwiseMode::CEIL:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_CEIL;
    case PointwiseMode::CMP_EQ:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_CMP_EQ;
    case PointwiseMode::CMP_GE:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_CMP_GE;
    case PointwiseMode::CMP_GT:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_CMP_GT;
    case PointwiseMode::CMP_LE:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_CMP_LE;
    case PointwiseMode::CMP_LT:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_CMP_LT;
    case PointwiseMode::CMP_NEQ:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_CMP_NEQ;
    case PointwiseMode::DIV:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_DIV;
    case PointwiseMode::ELU_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_ELU_BWD;
    case PointwiseMode::ELU_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_ELU_FWD;
    case PointwiseMode::ERF:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_ERF;
    case PointwiseMode::EXP:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_EXP;
    case PointwiseMode::FLOOR:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_FLOOR;
    case PointwiseMode::GELU_APPROX_TANH_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_GELU_APPROX_TANH_BWD;
    case PointwiseMode::GELU_APPROX_TANH_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_GELU_APPROX_TANH_FWD;
    case PointwiseMode::GELU_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_GELU_BWD;
    case PointwiseMode::GELU_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_GELU_FWD;
    case PointwiseMode::GEN_INDEX:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_GEN_INDEX;
    case PointwiseMode::IDENTITY:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_IDENTITY;
    case PointwiseMode::LOG:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_LOG;
    case PointwiseMode::LOGICAL_AND:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_LOGICAL_AND;
    case PointwiseMode::LOGICAL_NOT:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_LOGICAL_NOT;
    case PointwiseMode::LOGICAL_OR:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_LOGICAL_OR;
    case PointwiseMode::MAX:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_MAX_OP;
    case PointwiseMode::MIN:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_MIN_OP;
    case PointwiseMode::MUL:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_MUL;
    case PointwiseMode::NEG:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_NEG;
    case PointwiseMode::RECIPROCAL:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_RECIPROCAL;
    case PointwiseMode::RELU_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_RELU_BWD;
    case PointwiseMode::RELU_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_RELU_FWD;
    case PointwiseMode::RSQRT:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_RSQRT;
    case PointwiseMode::SIGMOID_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_SIGMOID_BWD;
    case PointwiseMode::SIGMOID_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_SIGMOID_FWD;
    case PointwiseMode::SIN:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_SIN;
    case PointwiseMode::SOFTPLUS_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_SOFTPLUS_BWD;
    case PointwiseMode::SOFTPLUS_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_SOFTPLUS_FWD;
    case PointwiseMode::SQRT:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_SQRT;
    case PointwiseMode::SUB:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_SUB;
    case PointwiseMode::SWISH_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_SWISH_BWD;
    case PointwiseMode::SWISH_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_SWISH_FWD;
    case PointwiseMode::TAN:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_TAN;
    case PointwiseMode::TANH_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_TANH_BWD;
    case PointwiseMode::TANH_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_TANH_FWD;
    default:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_UNSET;
    }
}

inline hipdnnBackendHeurMode_t toBackendType(const HeuristicMode& type)
{
    switch(type)
    {
    case HeuristicMode::FALLBACK:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    default:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    }
}

// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const DataType& type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return "fp32";
    case DataType::HALF:
        return "fp16";
    case DataType::BFLOAT16:
        return "bf16";
    case DataType::DOUBLE:
        return "fp64";
    case DataType::UINT8:
        return "uint8";
    case DataType::INT32:
        return "int32";
    default:
        return "unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, const DataType& type)
{
    return os << to_string(type);
}

inline const std::set<PointwiseMode>& unaryPointwiseModes()
{
    static const std::set<PointwiseMode> unaryPointwiseModes = {PointwiseMode::ABS,
                                                                PointwiseMode::CEIL,
                                                                PointwiseMode::ELU_FWD,
                                                                PointwiseMode::ERF,
                                                                PointwiseMode::EXP,
                                                                PointwiseMode::FLOOR,
                                                                PointwiseMode::GELU_APPROX_TANH_FWD,
                                                                PointwiseMode::GELU_FWD,
                                                                PointwiseMode::GEN_INDEX,
                                                                PointwiseMode::IDENTITY,
                                                                PointwiseMode::LOG,
                                                                PointwiseMode::LOGICAL_NOT,
                                                                PointwiseMode::NEG,
                                                                PointwiseMode::RECIPROCAL,
                                                                PointwiseMode::RELU_FWD,
                                                                PointwiseMode::RSQRT,
                                                                PointwiseMode::SIGMOID_FWD,
                                                                PointwiseMode::SIN,
                                                                PointwiseMode::SOFTPLUS_FWD,
                                                                PointwiseMode::SQRT,
                                                                PointwiseMode::SWISH_FWD,
                                                                PointwiseMode::TAN,
                                                                PointwiseMode::TANH_FWD};

    return unaryPointwiseModes;
}

inline const std::set<PointwiseMode>& binaryPointwiseModes()
{
    static const std::set<PointwiseMode> binaryPointwiseModes
        = {PointwiseMode::ADD,
           PointwiseMode::ADD_SQUARE,
           PointwiseMode::CMP_EQ,
           PointwiseMode::CMP_GE,
           PointwiseMode::CMP_GT,
           PointwiseMode::CMP_LE,
           PointwiseMode::CMP_LT,
           PointwiseMode::CMP_NEQ,
           PointwiseMode::DIV,
           PointwiseMode::ELU_BWD,
           PointwiseMode::GELU_APPROX_TANH_BWD,
           PointwiseMode::GELU_BWD,
           PointwiseMode::LOGICAL_AND,
           PointwiseMode::LOGICAL_OR,
           PointwiseMode::MAX,
           PointwiseMode::MIN,
           PointwiseMode::MUL,
           PointwiseMode::RELU_BWD,
           PointwiseMode::SIGMOID_BWD,
           PointwiseMode::SOFTPLUS_BWD,
           PointwiseMode::SUB,
           PointwiseMode::SWISH_BWD,
           PointwiseMode::TANH_BWD};

    return binaryPointwiseModes;
}

inline const std::set<PointwiseMode>& ternaryPointwiseModes()
{
    static const std::set<PointwiseMode> ternaryPointwiseModes = {PointwiseMode::BINARY_SELECT};

    return ternaryPointwiseModes;
}

}
