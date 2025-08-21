// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace hipdnn_sdk
{
namespace utilities
{

// NOLINTBEGIN(portability-template-virtual-member-function)

/// @brief Interface for host memory allocators
template <typename T>
class Host_allocator_interface
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = Host_allocator_interface<U>;
    };

    virtual ~Host_allocator_interface() = default;

    /// @brief Allocate memory
    /// @param n Number of elements to allocate
    /// @return Pointer to allocated memory
    /// @throws std::bad_alloc if allocation fails
    [[nodiscard]] virtual T* allocate(std::size_t n) = 0;

    /// @brief Deallocate memory
    /// @param p Pointer to memory to deallocate
    /// @param n Number of elements (may be used by some allocators)
    virtual void deallocate(T* p, std::size_t n) noexcept = 0;

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args)
    {
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void destroy(U* p)
    {
        p->~U();
    }
};

/// @brief Interface for device memory allocators
template <typename T>
class Device_allocator_interface
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = Device_allocator_interface<U>;
    };

    virtual ~Device_allocator_interface() = default;

    /// @brief Allocate device memory
    /// @param n Number of elements to allocate
    /// @return Pointer to allocated device memory
    /// @throws std::bad_alloc if allocation fails
    [[nodiscard]] virtual T* allocate(std::size_t n) = 0;

    /// @brief Deallocate device memory
    /// @param p Pointer to device memory to deallocate
    /// @param n Number of elements (may be used by some allocators)
    virtual void deallocate(T* p, std::size_t n) noexcept = 0;
};

// NOLINTEND(portability-template-virtual-member-function)

/// @brief Standard host allocator using malloc/free
template <typename T>
class Host_allocator : public Host_allocator_interface<T>
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = Host_allocator<U>;
    };

    Host_allocator() noexcept = default;
    Host_allocator(const Host_allocator&) noexcept = default;

    template <typename U>
    Host_allocator(const Host_allocator<U>& other) noexcept
    {
        std::ignore = other;
    }

    ~Host_allocator() override = default;

    [[nodiscard]] T* allocate(std::size_t n) override
    {
        if(n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }

        auto ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
        if(!ptr)
        {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* p, std::size_t n) noexcept override
    {
        std::ignore = n;
        std::free(p);
    }
};

template <typename T, typename U>
bool operator==(const Host_allocator<T>& lhs, const Host_allocator<U>& rhs) noexcept
{
    std::ignore = lhs;
    std::ignore = rhs;
    return true;
}

template <typename T, typename U>
bool operator!=(const Host_allocator<T>& lhs, const Host_allocator<U>& rhs) noexcept
{
    std::ignore = lhs;
    std::ignore = rhs;
    return false;
}

/// @brief Pinned host allocator using hipHostMalloc/hipHostFree
template <typename T>
class Pinned_host_allocator : public Host_allocator_interface<T>
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = Pinned_host_allocator<U>;
    };

    Pinned_host_allocator() noexcept = default;
    Pinned_host_allocator(const Pinned_host_allocator&) noexcept = default;

    template <typename U>
    Pinned_host_allocator(const Pinned_host_allocator<U>& other) noexcept
    {
        std::ignore = other;
    }

    ~Pinned_host_allocator() override = default;

    [[nodiscard]] T* allocate(std::size_t n) override
    {
        if(n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }

        void* ptr = nullptr;
        hipError_t err = hipHostMalloc(&ptr, n * sizeof(T));
        if(err != hipSuccess)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t n) noexcept override
    {
        std::ignore = n;
        std::ignore = hipHostFree(p);
    }
};

template <typename T, typename U>
bool operator==(const Pinned_host_allocator<T>& lhs, const Pinned_host_allocator<U>& rhs) noexcept
{
    std::ignore = lhs;
    std::ignore = rhs;
    return true;
}

template <typename T, typename U>
bool operator!=(const Pinned_host_allocator<T>& lhs, const Pinned_host_allocator<U>& rhs) noexcept
{
    std::ignore = lhs;
    std::ignore = rhs;
    return false;
}

/// @brief Device allocator using hipMalloc/hipFree
template <typename T>
class Device_allocator : public Device_allocator_interface<T>
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = Device_allocator<U>;
    };

    Device_allocator() noexcept = default;
    Device_allocator(const Device_allocator&) noexcept = default;

    template <typename U>
    Device_allocator(const Device_allocator<U>& other) noexcept
    {
        std::ignore = other;
    }

    ~Device_allocator() override = default;

    [[nodiscard]] T* allocate(std::size_t n) override
    {
        if(n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }

        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, n * sizeof(T));
        if(err != hipSuccess)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t n) noexcept override
    {
        std::ignore = n;
        std::ignore = hipFree(p);
    }

    // Note: construct and destroy are not provided for device allocator
    // as they would require host code to run on device memory addresses
    // which is not valid. Device memory should be initialized using kernels.
};

template <typename T, typename U>
bool operator==(const Device_allocator<T>& lhs, const Device_allocator<U>& rhs) noexcept
{
    std::ignore = lhs;
    std::ignore = rhs;
    return true;
}

template <typename T, typename U>
bool operator!=(const Device_allocator<T>& lhs, const Device_allocator<U>& rhs) noexcept
{
    std::ignore = lhs;
    std::ignore = rhs;
    return false;
}

} // namespace utilities
} // namespace hipdnn_sdk
