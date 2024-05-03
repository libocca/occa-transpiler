#pragma once

#include <oklt/core/error.h>

#include <clang/Lex/Token.h>

#include <tl/expected.hpp>

namespace clang {
class Preprocessor;
}

namespace oklt {
struct OklAttribute;

using OklAttrVisitor = std::function<
    bool(const OklAttribute&, const std::vector<clang::Token>&, clang::Preprocessor&)>;

tl::expected<void, Error> visitOklAttributes(const std::vector<clang::Token>& tokens,
                                             clang::Preprocessor& pp,
                                             OklAttrVisitor visitor);
}  // namespace oklt
