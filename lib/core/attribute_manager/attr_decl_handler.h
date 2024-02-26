#pragma once

#include <clang/AST/Attr.h>

#include <functional>
#include <tl/expected.hpp>
#include <any>
#include <oklt/core/error.h>

namespace oklt {

class SessionStage;

class AttrDeclHandler {
   public:
    using HandleType = std::function<tl::expected<std::any, Error>(const clang::Attr*,
                                                                   const clang::Decl*,
                                                    const std::any*,
                                                                   SessionStage&)>;

    explicit AttrDeclHandler(HandleType h)
        : _handler(std::move(h)) {}

    AttrDeclHandler(AttrDeclHandler&&) = default;
    ~AttrDeclHandler() = default;

    tl::expected<std::any, Error> handle(const clang::Attr* attr,
                                         const clang::Decl*,
                                                    const std::any* params,
                                         SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
