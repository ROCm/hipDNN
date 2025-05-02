// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#include <hip/hip_runtime_api.h>
#pragma clang diagnostic pop

// Cmake Generated export header.
#include "hipdnn_backend_export.h"

#include "hipdnn_backend_attribute_name.h"
#include "hipdnn_backend_attribute_type.h"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_status.h"

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

/*!
* @brief Creates the hipdnnHandle_t type
*/
typedef struct hipdnnHandle* hipdnnHandle_t;

/*!
* @brief Creates the hipdnnBackendDescriptor_t type
*/
typedef struct hipdnnBackendDescriptor* hipdnnBackendDescriptor_t;

/*! @brief Creates a hipdnnHandle_t
 *
 * @param [in] handle        An instance of hipdnnHandle_t
 *
 * @retval HIPDNN_STATUS_SUCCESS            The creation was successful
 * @retval HIPDNN_STATUS_BAD_PARAM          The descriptor is not a valid (NULL) descriptor.
 * @retval HIPDNN_STATUS_ALLOC_FAILED       The memory allocation failed when creating handle object.
 * 
 */
hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle);

/*! @brief Destroyes hipdnnHandle_t
 *
 * @param [in] handle        An instance of hipdnnHandle_t
 * 
 * @retval HIPDNN_STATUS_BAD_PARAM          The descriptor is not a valid (NULL) descriptor.
 * @retval HIPDNN_STATUS_SUCCESS            The destruction was successful
 * 
 */
hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle);

/*! @brief Sets the stream for the hipdnnHandle_t
 *
 * @param [in] handle        An instance of hipdnnHandle_t
 * @param [in] streamId      The stream to be set
 *
 * @retval HIPDNN_STATUS_BAD_PARAM                  invalid (NULL) handle.
 * @retval HIPDNN_STATUS_SUCCESS                    The creation was successful
 * 
 */
hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId);

/**
 * @brief Retrieves the HIP stream associated with the specified hipDNN handle.
 *
 * @param[in] handle The hipDNN handle whose associated stream is to be retrieved.
 * @param[out] streamId Pointer to a hipStream_t where the associated stream ID
 *                      will be stored upon successful execution.
 *
 * @retval HIPDNN_STATUS_BAD_PARAM                  invalid (NULL) handle.
 * @retval HIPDNN_STATUS_SUCCESS                    The creation was successful
 *
 * @note
 * - The handle must be valid and initialized before calling this function.
 * - The streamId pointer must not be null.
 * - This function uses a try-catch mechanism to handle exceptions internally.
 */
hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* streamId);

/*! @brief Creates a backend descriptor
 *
 * Allocates memory for a given descriptor_type at the location pointed
 * by the descriptor
 *
 * @param [in]   descriptor_type  One among the enumerated hipdnnBackendDescriptor_t
 * @param [out]  descriptor       Pointer to a descriptor
 *
 * @retval  HIPDNN_STATUS_SUCCESS            The creation was successful
 * @retval  HIPDNN_STATUS_NOT_SUPPORTED       Creating a descriptor of a given type is not supported.
 * @retval  HIPDNN_STATUS_ALLOC_FAILED        The memory allocation failed.
 * 
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateDescriptor(
    hipdnnBackendDescriptorType_t descriptor_type, hipdnnBackendDescriptor_t* descriptor);

/*! @brief Destroys an instance of hipdnnBackendDescriptor_t
 *
 * Destroys instances of hipdnnBackendDescriptor_t that were previously created using
 * hipdnnBackendCreateDescriptor(). The value pointed by the descriptor will be undefined after the
 * memory is free and done.
 *
 * **Undefined Behavior** if the descriptor was altered between the 'Create' and 'Destroy
 * Descriptor'
 *
 * @param  [in]  descriptor  Instance of hipdnnBackendDescriptor_t previously created by
 *                           hipdnnBackendCreateDescriptor()
 *
 * @retval  HIPDNN_STATUS_SUCCESS       The memory was destroyed successfully
 * @retval  HIPDNN_STATUS_ALLOC_FAILED   The destruction of memory failed.
 * 
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor);

/*! @brief Executes a graph
 *
 * Executes the given engine_configuration_plan on the variant_pack and the finalized execution_plan
 * on the data. The data and the working space are encapsulated in the variant_pack.
 *
 * @param  [in]  handle              An instance of hipdnnHandle_t
 * @param  [in]  execution_plan      Descriptor of the finalized execution_plan
 * @param  [in]  variant_pack        Descriptor of the finalized variant_pack consisting of:
 *                                   * Data pointer for each non-virtual pointer of the operation set in
 *                                     the execution plan.
 *                                   * Pointer to user-allocated workspace in global memory at least as
 *                                     large as the size queried
 *
 * @retval  HIPDNN_STATUS_SUCCESS               The execution_plan was executed successfully
 * @retval  HIPDNN_STATUS_BAD_PARAM             An incorrect or inconsistent value is encountered. For example, a required data pointer is invalid.
 * @retval  HIPDNN_STATUS_INTERNAL_ERROR        Some internal errors were encountered.
 * @retval  HIPDNN_STATUS_EXECUTION_FAILED      An error was encountered executing the plan with the variant_pack.
 * 
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendExecute(hipdnnHandle_t handle,
                                                          hipdnnBackendDescriptor_t execution_plan,
                                                          hipdnnBackendDescriptor_t variant_pack);

/*! @brief Finalizes a backend descriptor
*
* Finalizes the memory pointed to by the descriptor. The type of finalization is done depending on
* the descriptor_type argument with which the descriptor was created using
* hipdnnBackendCreateDescriptor() or initialized using hipdnnBackendInitialize().
*
* @param  [in]  descriptor  Instance of hipdnnBackendDescriptor_t to finalize
*
* @retval  HIPDNN_STATUS_SUCCESS           The descriptor was finalized successfully
* @retval  HIPDNN_STATUS_BAD_PARAM         Invalid descriptor attribute values or combination thereof is encountered.
* @retval  HIPDNN_STATUS_NOT_SUPPORTED     Descriptor attribute values or combinations therefore not supported by the current version of hipDNN are encountered.
* @retval  HIPDNN_STATUS_INTERNAL_ERROR    Some internal errors are encountered.

*/
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor);

/*! @brief Retrieves backend descriptor's attribute
 *
 * This function retrieves the values of an attribute of a descriptor. attribute_name is the name of
 * the attribute whose value is requested. attribute_type is the type of attribute.
 * requested_element_count is the number of elements to be potentially retrieved. The number of
 * elements for the requested attribute is stored in element_count. The retrieved values are stored
 * in array_of_elements. When the attribute is expected to have a single value, array_of_elements can be
 * pointer to the output value. This function will return HIPDNN_STATUS_NOT_INTIALIZED if the
 * descriptor has not been successfully finalized using hipdnnBackendFinalize()
 *
 * @param  [in]   descriptor               Instance of hipdnnBackendDescriptor_t whose attribute to
 *                                         retrieve
 * @param  [in]   attribute_name           The name of the attribute being get from the descriptor
 * @param  [in]   attribute_type           The type of attribute
 * @param  [in]   requested_element_count  Number of elements to output to array_of_elements
 * @param  [out]  element_count            Output pointer for the number of elements the descriptor
 *                                         attribute has. Note that hipdnnBackendGetAttribute() will
 *                                         only write the least of this and requested_element_count
 *                                         elements to array_of_elements
 * @param  [out]  array_of_elements        Array of elements of the datatype of the attribute_type. The
 *                                         data type of the attribute_type is listed in the mapping
 *                                         table of hipdnnBackendAttributeType_t
 *
 * @retval  HIPDNN_STATUS_SUCCESS           The attribute_name was retrieved from the descriptor successfully
 * @retval  HIPDNN_STATUS_BAD_PARAM         One or more invalid or inconsistent argument values were encountered. Some examples include: 
 *                                              attribute_name is not a valid attribute for the descriptor. 
 *                                              attribute_type is not one of the valid types for the attribute.
 * @retval  HIPDNN_STATUS_NOT_INITIALIZED   The descriptor has not been successfully finalized using hipdnnBackendFinalize().
 * 
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attribute_name,
                              hipdnnBackendAttributeType_t attribute_type,
                              int64_t requested_element_count,
                              int64_t* element_count,
                              void* array_of_elements);

/*! @brief Sets an attribute of a descriptor
 *
 * This function sets an attribute of a descriptor to values provided as a pointer. descriptor is the descriptor to be set. 
 * attribute_name is the name of the attribute to be set. attribute_type is the type of attribute. 
 * The value to which the attribute is set, is pointed by the array_of_elements. The number of elements is given by element_count. 
 * This function will return HIPDNN_STATUS_NOT_INITIALIZED if the descriptor is already successfully finalized using hipdnnBackendFinalize().
 *
 * @param  [in]  descriptor         Instance of hipdnnBackendDescriptor_t whose attribute is being set
 * @param  [in]  attribute_name     The name of the attribute being set on the descriptor
 * @param  [in]  attribute_type     The type of attribute
 * @param  [in]  element_count      Number of elements being set
 * @param  [in]  array_of_elements  The starting location for an array from where to read the values
 *                                  from. The elements of the array are expected to be of the datatype
 *                                  of the attribute_type. The datatype of the attribute_type is listed
 *                                  in the mapping table of hipdnnBackendAttributeType_t.
 *
 * @retval  HIPDNN_STATUS_SUCCESS           The attribute_name was set to the descriptor
 * @retval  HIPDNN_STATUS_NOT_INITIALIZED   The backend descriptor pointed to by the descriptor is already in the finalized state.
 * @retval  HIPDNN_STATUS_BAD_PARAM         The function is called with arguments that correspond to invalid values. Some examples include:
 *                                              attribute_name is not a settable attribute of descriptor.
 *                                              attribute_type is incorrect for this attribute_name.
 *                                              element_count value is unexpected.
 *                                              array_of_elements contains values invalid for the attribute_type.
 * @retval  HIPDNN_STATUS_NOT_SUPPORTED     The values to which the attributes are being set are not supported by the current version of hipDNN.
 * 
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attribute_name,
                              hipdnnBackendAttributeType_t attribute_type,
                              int64_t element_count,
                              const void* array_of_elements);

/*
 **************************************************************************************************
 *
 *  Extension API Below
 *
 **************************************************************************************************
 */

/*!
 * @brief Creates and deserializes a graph into a backend descriptor.
 *
 * This function creates a backend descriptor and deserializes a graph from a serialized byte array 
 * into the descriptor. The serialized graph is provided as an input byte array, and the size of 
 * the graph in bytes is specified. The created descriptor will encapsulate the deserialized graph.
 *
 * @param [out] descriptor        Pointer to a backend descriptor where the deserialized graph will 
 *                                be stored.
 * @param [in]  serialized_graph  Pointer to the serialized graph data in a byte array.
 * @param [in]  graph_byte_size   Size of the serialized graph in bytes.
 *
 * @retval HIPDNN_STATUS_SUCCESS           The graph was successfully deserialized and stored in the descriptor.
 * @retval HIPDNN_STATUS_BAD_PARAM         Invalid or inconsistent parameter values were encountered, such as:
 *                                         - descriptor is null.
 *                                         - serialized_graph is null.
 *                                         - graph_byte_size is zero.
 * @retval HIPDNN_STATUS_ALLOC_FAILED      Memory allocation for the descriptor or graph failed.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR    An internal error occurred during deserialization.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const uint8_t* serialized_graph, size_t graph_byte_size);

#ifdef __cplusplus
}
#endif
// NOLINTEND