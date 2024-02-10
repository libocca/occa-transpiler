#pragma once

#include <tl/expected.hpp>
#include <clang/AST/Attr.h>
#include <clang/Basic/LLVM.h>
#include <oklt/core/error.h>
#include <functional>

namespace oklt {

class SessionStage;
//INFO: if needed to achive different behaviour
//      add your own validator and
//      just pass to your implementation of Semantic Analyzer

using ValidatorResult = tl::expected<const clang::Attr*, Error>;
ValidatorResult validateAttributes(const clang::ArrayRef<const clang::Attr *> &attrs,
                                   SessionStage &stage);

using AttrValidatorFnType = std::function<ValidatorResult(const clang::ArrayRef<const clang::Attr *> &attrs,
                                                          SessionStage &stage)>;

}
