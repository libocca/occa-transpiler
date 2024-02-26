#pragma once

#include <clang/AST/Attr.h>

#include <functional>

namespace oklt {

class SessionStage;

class AttrStmtHandler {
   public:
    using HandleType = std::function<tl::expected<std::any, Error>(const clang::Attr*,
                                                                   const clang::Stmt*,
                                                    const std::any*,
                                                                   SessionStage&)>;

    explicit AttrStmtHandler(HandleType h)
        : _handler(std::move(h)) {}

    AttrStmtHandler(AttrStmtHandler&&) = default;
    ~AttrStmtHandler() = default;

    tl::expected<std::any, Error> handle(const clang::Attr* attr,
                                         const clang::Stmt*,
                                                    const std::any* params,
                                         SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
