#pragma once

#include "okl_attribute.h"

#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/Token.h>

namespace oklt {

using OklAttrVisitor = std::function<
    bool(const OklAttribute&, const std::vector<clang::Token>&, clang::Preprocessor&)>;

int visitOklAttributes(const std::vector<clang::Token>& tokens,
                       clang::Preprocessor& pp,
                       OklAttrVisitor visitor);
}  // namespace oklt
