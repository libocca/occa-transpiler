#pragma once

#include <clang/AST/Attr.h>

#include <functional>
#include <any>
#include <tl/expected.hpp>
#include <oklt/core/error.h>

namespace oklt {

class SessionStage;

class DeclHandler {
   public:
    using HandleType = std::function<tl::expected<std::any, Error>(const clang::Decl*, SessionStage&)>;

    explicit DeclHandler(HandleType h);
    ~DeclHandler() = default;

    tl::expected<std::any, Error> operator()(const clang::Decl*, SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
