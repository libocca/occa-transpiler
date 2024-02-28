#pragma once

#include "util/type_traits.h"

#include <any>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace clang {
class Attr;
}

namespace oklt {

class SessionStage;
class OKLParsedAttr;

class OKLAttrParam {
   public:
    explicit OKLAttrParam(std::string_view raw, std::any v)
        : rawData(raw),
          data(std::move(v)){};

    /// @brief return raw string representation
    [[nodiscard]] std::string_view getRaw() const { return rawData; }

    /// @brief Checks if attribute is empty or not
    [[nodiscard]] bool empty();

    /// @brief check if value is integer
    [[nodiscard]] bool is_integral();
    /// @brief check if value is unsigned integer
    [[nodiscard]] bool is_unsigned();
    /// @brief check if value is a floating point
    [[nodiscard]] bool is_float();
    /// @brief check if value is a string
    [[nodiscard]] bool is_string();
    /// @brief check if value is an OKL attribute
    [[nodiscard]] bool is_attr();
    /// @brief check if value is an expression
    [[nodiscard]] bool is_expr();

    template <typename T, typename std::enable_if_t<std::is_integral_v<T>, bool> = true>
    [[nodiscard]] bool isa();

    template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    [[nodiscard]] bool isa();

    template <typename T, typename std::enable_if_t<is_string_v<T>, bool> = true>
    [[nodiscard]] bool isa();

    template <typename T, typename std::enable_if_t<std::is_same_v<T, OKLParsedAttr>, bool> = true>
    [[nodiscard]] bool isa();

    //    /// @brief check if value is of type
    //    template <typename T>
    //    [[nodiscard]] bool isa() const {
    //        return false;
    //    }

    /// @brief check if value if of given types
    template <typename F, typename S, typename... T>
    [[nodiscard]] inline bool isa() {
        return isa<F>() || isa<S, T...>();
    }

    template <typename T, typename std::enable_if_t<std::is_integral_v<T>, bool> = true>
    [[nodiscard]] std::optional<T> get();

    template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    [[nodiscard]] std::optional<T> get();

    template <typename T, typename std::enable_if_t<is_string_v<T>, bool> = true>
    [[nodiscard]] std::optional<T> get();

    template <typename T,
        typename std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, OKLParsedAttr>, bool> = true>
    [[nodiscard]] std::optional<T> get();

    //    /// @brief get value of desired type or std::nullopt
    //    template <typename T>
    //    [[nodiscard]] std::optional<T> get() const {
    //        return std::nullopt;
    //    }

    /// @brief get value to referenced buffer, return true on success.
    template <typename T>
    bool getTo(T& v);

    template <typename T>
    void getTo(T& v, T&& u);

   private:
    std::string_view rawData;
    std::any data;
};

struct OKLParsedAttr {
    explicit OKLParsedAttr();
    explicit OKLParsedAttr(const std::string_view name);

    std::string name;

    std::vector<OKLAttrParam> args;
    template <typename T = OKLAttrParam>
    [[nodiscard]] std::optional<T> get(size_t n);
    template <typename T = OKLAttrParam>
    [[nodiscard]] T get(size_t n, T&& u);
    template <typename... T>
    [[nodiscard]] bool isa(size_t n);

    std::map<std::string_view, OKLAttrParam> kwargs;
    template <typename T = OKLAttrParam>
    [[nodiscard]] std::optional<T> get(std::string_view k);
    template <typename T = OKLAttrParam>
    [[nodiscard]] T get(std::string_view k, T&& u);
    template <typename... T>
    [[nodiscard]] bool isa(std::string_view k);
};

OKLParsedAttr ParseOKLAttr(const clang::Attr& attr, SessionStage& stage);

}  // namespace oklt
