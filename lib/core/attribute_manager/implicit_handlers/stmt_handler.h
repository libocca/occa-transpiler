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
    using HandleType = std::function<HandleResult(SessionStage&, const clang::Stmt&)>;

    explicit StmtHandler(HandleType h);
    ~StmtHandler() = default;

    HandleResult operator()(SessionStage& stage, const clang::Stmt&);

   private:
    HandleType _handler;
};
}  // namespace oklt
