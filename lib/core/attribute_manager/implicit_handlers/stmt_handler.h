#pragma once

#include <clang/AST/Attr.h>

#include <functional>
#include <tl/expected.hpp>
#include <any>
#include <oklt/core/error.h>

namespace oklt {

class SessionStage;

class StmtHandler {
   public:
    using HandleType =
        std::function<tl::expected<std::any, Error>(const clang::Stmt*, SessionStage&)>;

    explicit StmtHandler(HandleType h);
    ~StmtHandler() = default;

    tl::expected<std::any, Error> operator()(const clang::Stmt*, SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
