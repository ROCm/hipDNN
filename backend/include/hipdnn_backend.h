#pragma once

#define HIPDNN_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

    HIPDNN_EXPORT int publicFunctionHello();

#ifdef __cplusplus
}
#endif