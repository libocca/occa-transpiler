#pragma once

#include "attributes/utils/parser.h"
#include "parser.h"
#include "util/type_traits.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APSInt.h>

#include <memory>
#include <optional>
#include <utility>

namespace oklt {

inline bool OKLAttrParam::empty() {
    return rawData.empty();
}

inline bool OKLAttrParam::is_integral() {
    return (data.has_value() && data.type() == typeid(llvm::APSInt));
}

inline bool OKLAttrParam::is_unsigned() {
    if (!is_integral())
        return false;

    auto& val = std::any_cast<llvm::APSInt&>(data);
    return val.isUnsigned();
}

inline bool OKLAttrParam::is_float() {
    return (data.has_value() && data.type() == typeid(llvm::APFloat));
}

inline bool OKLAttrParam::is_string() {
    return (data.has_value() &&
            (data.type() == typeid(std::string) || data.type() == typeid(std::wstring) ||
             data.type() == typeid(std::u16string) || data.type() == typeid(std::u32string)));
}

inline bool OKLAttrParam::is_attr() {
    return (data.has_value() && data.type() == typeid(OKLParsedAttr));
}

inline bool OKLAttrParam::is_expr() {
    return (!data.has_value() && !rawData.empty());
}

template <typename T, typename std::enable_if_t<std::is_integral_v<T>, bool>>
bool OKLAttrParam::isa() {
    if (!data.has_value() || data.type() != typeid(llvm::APSInt))
        return false;

    auto& val = std::any_cast<llvm::APSInt&>(data);
    if (val.getBitWidth() > sizeof(T))
        return false;
    if (val.isSigned() != std::numeric_limits<T>::is_signed)
        return false;

    return true;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool>>
bool OKLAttrParam::isa() {
    if (!data.has_value() || data.type() != typeid(llvm::APFloat))
        return false;

    auto& val = std::any_cast<llvm::APFloat&>(data);
    if (llvm::APFloat::getSizeInBits(val.getSemantics()) != sizeof(T) * 8)
        return false;

    return true;
}

template <typename T, std::enable_if_t<is_string_v<T>, bool>>
bool OKLAttrParam::isa() {
    using cell_t = typename T::value_type;
    return (data.has_value() && (data.type() == typeid(std::string) || isa<cell_t>()));
}

template <typename T, std::enable_if_t<std::is_same_v<T, OKLParsedAttr>, bool>>
bool OKLAttrParam::isa() {
    return (data.has_value() && data.type() == typeid(T));
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool>>
std::optional<T> OKLAttrParam::get() {
    if (!data.has_value() || data.type() != typeid(llvm::APSInt))
        return std::nullopt;  // Not an integer

    auto& val = std::any_cast<llvm::APSInt&>(data);
    if (val < std::numeric_limits<T>::min() || val > std::numeric_limits<T>::max())
        return std::nullopt;

    T ret = *reinterpret_cast<const T*>(val.abs().getRawData());
    if (val.isNegative())
        ret = -ret;
    return ret;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool>>
std::optional<T> OKLAttrParam::get() {
    if (!data.has_value() || data.type() != typeid(llvm::APFloat))
        return std::nullopt;  // Not an integer

    auto& val = std::any_cast<llvm::APFloat&>(data);

    if (llvm::APFloat::getSizeInBits(val.getSemantics()) > sizeof(T) * 8)
        return std::nullopt;  // Downcast

    if constexpr (std::is_same_v<T, float>)
        return val.convertToFloat();

    if constexpr (std::is_same_v<T, double>)
        return val.convertToDouble();

    // NOTE: There is no native way to extract
    if constexpr (std::is_same_v<T, long double>) {
        std::array<char, 256> cstr = {};
        val.convertToHexString(cstr.data(), 0, false, llvm::APFloat::rmNearestTiesToEven);
        return std::strtold(cstr.data(), nullptr);
    }

    return std::nullopt;
}

template <typename T, std::enable_if_t<is_string_v<T>, bool>>
std::optional<T> OKLAttrParam::get() {
    using cell_t = typename T::value_type;
    if (!data.has_value())
        return std::nullopt;

    if (data.type() == typeid(T))
        return std::any_cast<T>(data);

    if (data.type() == typeid(llvm::APSInt)) {
        auto t = get<cell_t>();
        if (t.has_value())
            return T(t.value(), 1);
    }

    return std::nullopt;
}

template <typename T, std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, OKLParsedAttr>, bool>>
std::optional<T> OKLAttrParam::get() {
    if (!data.has_value() || data.type() != typeid(OKLParsedAttr))
        return std::nullopt;  // Not an OKLParsedAttr

    return std::any_cast<OKLParsedAttr>(data);
}

template <typename T>
bool OKLAttrParam::getTo(T& v) {
    auto val = get<T>();
    if (val.has_value()) {
        v = std::move(val.value());
        return true;
    }

    return false;
}

template <typename T>
void OKLAttrParam::getTo(T& v, T&& u) {
    v = get<T>().value_or(u);
}

template <typename T>
std::optional<T> OKLParsedAttr::get(size_t n) {
    if (n < args.size()) {
        if constexpr (std::is_same_v<T, OKLAttrParam>)
            return args[n];
        return args[n].get<T>();
    }
    return std::nullopt;
};

template <typename T>
T OKLParsedAttr::get(size_t n, T&& u) {
    if (n < args.size()) {
        if constexpr (std::is_same_v<T, OKLAttrParam>)
            return args[n];
        return args[n].get<T>().value_or(u);
    }
    return u;
};

template <typename... T>
bool OKLParsedAttr::isa(size_t n) {
    if (n < args.size())
        return args[n].isa<T...>();
    return false;
};

template <typename T>
std::optional<T> OKLParsedAttr::get(std::string_view k) {
    auto it = kwargs.find(k);
    if (it != kwargs.end()) {
        if constexpr (std::is_same_v<T, OKLAttrParam>)
            return it->second;
        return it->second.get<T>();
    }
    return std::nullopt;
};

template <typename T>
T OKLParsedAttr::get(std::string_view k, T&& u) {
    auto it = kwargs.find(k);
    if (it != kwargs.end()) {
        if constexpr (std::is_same_v<T, OKLAttrParam>)
            return it->second;
        return it->second.get<T>().value_or(u);
    }
    return u;
};

template <typename... T>
bool OKLParsedAttr::isa(std::string_view k) {
    auto it = kwargs.find(k);
    if (it != kwargs.end())
        return it->second.isa<T...>();
    return false;
};

}  // namespace oklt
