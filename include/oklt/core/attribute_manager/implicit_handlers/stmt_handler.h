#pragma once

#include <clang/AST/Attr.h>
#include <functional>

namespace oklt {

class SessionStage;

class StmtHandler {
   public:
    using HandleType = std::function<bool(const clang::Stmt*, SessionStage&)>;

    explicit StmtHandler(HandleType h);
    ~StmtHandler() = default;

    bool operator()(const clang::Stmt*, SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
