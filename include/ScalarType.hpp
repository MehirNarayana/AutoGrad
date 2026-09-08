#pragma once

#include <cstdint>
#include <type_traits>

enum class DType : uint8_t { Float32 = 0, Double = 1, Int32 = 2, Int64 = 3 };

template <typename type>
inline constexpr bool isSupportedFloatingPointScalarType =
    std::is_same_v<std::remove_cv_t<type>, float> || std::is_same_v<std::remove_cv_t<type>, double>;

template <typename type>
inline constexpr bool isSupportedIntegralScalarType =
    std::is_same_v<std::remove_cv_t<type>, std::int32_t> ||
    std::is_same_v<std::remove_cv_t<type>, std::int64_t>;

template <typename type>
inline constexpr bool isSupportedTensorScalarType =
    isSupportedFloatingPointScalarType<type> || isSupportedIntegralScalarType<type>;

template <typename type>
inline constexpr bool isSupportedAutogradScalarType = isSupportedFloatingPointScalarType<type>;

template <typename type>
inline constexpr bool isSupportedIndexScalarType = isSupportedIntegralScalarType<type>;

template <typename type>
inline constexpr DType dType() {
    using cleanType = std::remove_cv_t<type>;

    if constexpr (std::is_same_v<cleanType, float>) {
        return DType::Float32;
    } else if constexpr (std::is_same_v<cleanType, double>) {
        return DType::Double;
    } else if constexpr (std::is_same_v<cleanType, std::int32_t>) {
        return DType::Int32;
    } else {
        return DType::Int64;
    }
}
