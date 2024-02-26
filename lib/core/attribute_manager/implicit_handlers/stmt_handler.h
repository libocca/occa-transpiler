#pragma once

#include "core/attribute_manager/result.h"

#include <clang/AST/Attr.h>

#include <oklt/core/error.h>
#include <any>
#include <functional>
#include <tl/expected.hpp>

namespace oklt {

class SessionStage;

class StmtHandler {
   public:
    using HandleType = std::function<HandleResult(const clang::Stmt*, SessionStage&)>;

    explicit StmtHandler(HandleType h);
    ~StmtHandler() = default;

    HandleResult operator()(const clang::Stmt*, SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
