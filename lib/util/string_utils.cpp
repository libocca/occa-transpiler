#include "oklt/util/string_utils.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <sstream>

namespace oklt::util {
using namespace llvm;

std::string toLower(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    std::transform(str.begin(), str.end(), std::back_inserter(result), ::tolower);
    return result;
}

std::string toCamelCase(std::string str) {
    std::size_t res_ind = 0;
    for (int i = 0; i < str.length(); i++) {
        // check for spaces in the sentence
        if (str[i] == ' ' || str[i] == '_') {
            // conversion into upper case
            str[i + 1] = ::toupper(str[i + 1]);
            continue;
        }
        // If not space, copy character
        else {
            str[res_ind++] = str[i];
        }
    }
    // return string to main
    return str.substr(0, res_ind);
}

std::string pointerToStr(const void* ptr) {
    return std::to_string(reinterpret_cast<uintptr_t>(ptr));
}

std::string_view rtrim(std::string_view& str) {
    return StringRef(str).rtrim();
}

std::string_view ltrim(std::string_view& str) {
    return StringRef(str).ltrim();
}

std::string_view trim(std::string_view& str) {
    return StringRef(str).trim();
}

std::string_view unParen(std::string_view& str) {
    if (!str.empty() && str.front() == '(' && str.back() == ')') {
        return str.substr(1, str.size() - 1);
    }
    return str;
}

std::string_view slice(const std::string_view& str, size_t start, size_t end) {
    return StringRef(str).slice(start, end);
}

std::vector<std::string_view> split(const std::string_view& str,
                                    const std::string_view& sep,
                                    int maxN,
                                    bool keepEmpty) {
    std::vector<std::string_view> ret;

    StringRef s(str);
    while (maxN-- != 0) {
        auto Idx = s.find(sep);
        if (Idx == StringRef::npos) {
            break;
        }
        if (keepEmpty || Idx > 0) {
            ret.push_back(s.slice(0, Idx));
        }
        s = s.slice(Idx + 1, StringRef::npos);
    }

    if (keepEmpty || !s.empty()) {
        ret.push_back(s);
    }

    return ret;
}

namespace impl {
tl::expected<size_t, Error> getCurlyBracketIdx(const std::string_view& str) {
    for (size_t idx = 0; idx < str.size(); ++idx) {
        if (str[idx] == fmtBrackets[0]) {
            if (idx >= (str.size() - 1) || str[idx + 1] != fmtBrackets[1]) {
                return tl::make_unexpected(Error{std::error_code(), "fmt: Bad formatting"});
            }
            return idx;
        }
    }
    return static_cast<size_t>(-1);
}

}  // namespace impl
tl::expected<std::string, Error> fmt(const std::string& s) {
    auto pos = impl::getCurlyBracketIdx(s);
    if (!pos.has_value()) {
        return tl::make_unexpected(pos.error());
    }

    if (pos.value() != static_cast<size_t>(-1)) {
        return tl::make_unexpected(
            Error{std::error_code(), "fmt: Not enough values (dangling '{}')"});
    }

    return s;
}

}  // namespace oklt::util
