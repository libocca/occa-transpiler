#pragma once

#include "core/attribute_manager/result.h"

#include <clang/AST/Attr.h>

#include <functional>

namespace oklt {

class SessionStage;

class AttrStmtHandler {
   public:
    using HandleType = std::function<
        HandleResult(SessionStage&, const clang::Stmt&, const clang::Attr&, const std::any*)>;

    explicit AttrStmtHandler(HandleType h)
        : _handler(std::move(h)) {}

    AttrStmtHandler(AttrStmtHandler&&) = default;
    ~AttrStmtHandler() = default;

    HandleResult handle(SessionStage& stage,
                        const clang::Stmt&,
                        const clang::Attr& attr,
                        const std::any* params);

   private:
    HandleType _handler;
};
}  // namespace oklt
