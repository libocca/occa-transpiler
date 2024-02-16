#pragma once

#include <clang/AST/Attr.h>
#include <functional>

namespace oklt {

class SessionStage;

class DeclHandler {
   public:
    using HandleType = std::function<bool(const clang::Decl*, SessionStage&)>;

    explicit DeclHandler(HandleType h);
    ~DeclHandler() = default;

    bool operator()(const clang::Decl*, SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
