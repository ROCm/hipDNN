// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
/**
 * @enum hipdnnEngineIds_t
 * @brief Enumeration for HIPDNN engine identifiers.
 * 
 * - HIPDNN_ENGINE_ID_FAKE: Represents a fake engine ID, used for mocking.
 * - HIPDNN_ENGINE_ID_MIOPEN: Represents the MIOpen engine ID, starting at 0.
 * - HIPDNN_ENGINE_ID_MIOPEN_Conv: Represents the MIOpen convolution engine ID, starting at 1.
 * - HIPDNN_ENGINE_ID_Custom_Plugin: Marks the beginning of the custom plugin range, starting at 1,000,000.
 * 
 * Note: Plugin IDs should be assigned ranges, e.g., MIOpen (0-100), IREE (101-200), etc.
 *
 * @brief Identifies different backend engines for the hipDNN library.
 * 
 * - HIPDNN_ENGINE_ID_FAKE: Placeholder or invalid engine.
 * - HIPDNN_ENGINE_ID_MIOPEN: Default backend engine.
 * - HIPDNN_ENGINE_ID_MIOPEN_Conv: Backend engine for MIOpen convolution operations.
 * - HIPDNN_ENGINE_ID_Custom_Plugin: Starting ID for custom plugin engines.
 */
typedef enum hipdnnEngineIds
{
    HIPDNN_ENGINE_ID_FAKE = -1,
    HIPDNN_ENGINE_ID_MIOPEN = 0,
    HIPDNN_ENGINE_ID_MIOPEN_Conv = 1,
    HIPDNN_ENGINE_ID_Custom_Plugin = 1000000

    // for now max value 2,147,483,647 (int max)

} hipdnnEngineIds_t;