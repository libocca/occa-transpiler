#pragma once

#include <clang/AST/Attr.h>
#include <functional>

namespace oklt {

class SessionStage;

class AttrStmtHandler {
   public:
    using HandleType = std::function<bool(const clang::Attr*, const clang::Stmt*, SessionStage&)>;

    explicit AttrStmtHandler(HandleType h)
        : _handler(std::move(h)) {}

    AttrStmtHandler(AttrStmtHandler&&) = default;
    ~AttrStmtHandler() = default;

    bool handle(const clang::Attr* attr, const clang::Stmt*, SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
