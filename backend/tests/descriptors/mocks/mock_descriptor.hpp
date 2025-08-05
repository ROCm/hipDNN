// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "descriptors/backend_descriptor.hpp"
#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"

#include <gmock/gmock.h>

namespace hipdnn_backend
{

class Mock_descriptor_utility
{
public:
    template <typename Child_descriptor>
    static std::shared_ptr<Child_descriptor>
        as_descriptor_unsafe(hipdnnBackendDescriptor_t descriptor)
    {
        if(!descriptor)
        {
            return nullptr;
        }

        return std::static_pointer_cast<Child_descriptor>(descriptor->_impl);
    }
};

template <typename Desc_type>
class Mock_descriptor : public hipdnnBackendDescriptorImpl<Mock_descriptor<Desc_type>>
{
public:
    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, is_finalized, (), (const, override));
    MOCK_METHOD(void,
                set_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t element_count,
                 const void* array_of_element),
                (override));
    MOCK_METHOD(void,
                get_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t requested_element_count,
                 int64_t* element_count,
                 void* array_of_elements),
                (const, override));

    static hipdnnBackendDescriptorType_t get_static_type()
    {
        return Desc_type::get_static_type();
    }
};

class Mock_engine_descriptor : public Engine_descriptor
{
public:
    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, is_finalized, (), (const, override));
    MOCK_METHOD(void,
                set_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t element_count,
                 const void* array_of_element),
                (override));
    MOCK_METHOD(void,
                get_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t requested_element_count,
                 int64_t* element_count,
                 void* array_of_elements),
                (const, override));

    MOCK_METHOD(std::shared_ptr<const Graph_descriptor>, get_graph, (), (const, override));
    MOCK_METHOD(int64_t, get_engine_id, (), (const, override));

    static hipdnnBackendDescriptorType_t get_static_type()
    {
        return HIPDNN_BACKEND_ENGINE_DESCRIPTOR;
    }
};

class Mock_engine_config_descriptor : public Engine_config_descriptor
{
public:
    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, is_finalized, (), (const, override));
    MOCK_METHOD(void,
                set_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t element_count,
                 const void* array_of_element),
                (override));
    MOCK_METHOD(void,
                get_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t requested_element_count,
                 int64_t* element_count,
                 void* array_of_elements),
                (const, override));

    MOCK_METHOD(std::shared_ptr<const Engine_descriptor>, get_engine, (), (const, override));
    MOCK_METHOD(hipdnnPluginConstData_t, get_serialized_engine_config, (), (const, override));

    static hipdnnBackendDescriptorType_t get_static_type()
    {
        return HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR;
    }
};

class Mock_graph_descriptor : public Graph_descriptor
{
public:
    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, is_finalized, (), (const, override));
    MOCK_METHOD(void,
                set_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t element_count,
                 const void* array_of_element),
                (override));
    MOCK_METHOD(void,
                get_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t requested_element_count,
                 int64_t* element_count,
                 void* array_of_elements),
                (const, override));

    MOCK_METHOD(hipdnnHandle_t, get_handle, (), (const, override));

    static hipdnnBackendDescriptorType_t get_static_type()
    {
        return HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
    }
};

ACTION_P(SetArg4ToInt64, value) // NOLINT
{
    *static_cast<int64_t*>(arg4) = value;
}

} // namespace hipdnn_backend
