
#ifndef HIPDNN_BACKEND_EXPORT_H
#define HIPDNN_BACKEND_EXPORT_H

#ifdef HIPDNN_BACKEND_STATIC_DEFINE
#define HIPDNN_BACKEND_EXPORT
#define HIPDNN_BACKEND_NO_EXPORT
#else
#ifndef HIPDNN_BACKEND_EXPORT
#ifdef hipdnn_backend_EXPORTS
/* We are building this library */
#define HIPDNN_BACKEND_EXPORT __attribute__((visibility("default")))
#else
/* We are using this library */
#define HIPDNN_BACKEND_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifndef HIPDNN_BACKEND_NO_EXPORT
#define HIPDNN_BACKEND_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif

#ifndef HIPDNN_BACKEND_DEPRECATED
#define HIPDNN_BACKEND_DEPRECATED __attribute__((__deprecated__))
#endif

#ifndef HIPDNN_BACKEND_DEPRECATED_EXPORT
#define HIPDNN_BACKEND_DEPRECATED_EXPORT HIPDNN_BACKEND_EXPORT HIPDNN_BACKEND_DEPRECATED
#endif

#ifndef HIPDNN_BACKEND_DEPRECATED_NO_EXPORT
#define HIPDNN_BACKEND_DEPRECATED_NO_EXPORT HIPDNN_BACKEND_NO_EXPORT HIPDNN_BACKEND_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#ifndef HIPDNN_BACKEND_NO_DEPRECATED
#define HIPDNN_BACKEND_NO_DEPRECATED
#endif
#endif

#endif /* HIPDNN_BACKEND_EXPORT_H */
