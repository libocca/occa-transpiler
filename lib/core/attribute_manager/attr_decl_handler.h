#pragma once

#include "core/attribute_manager/result.h"

#include <clang/AST/Attr.h>
#include <tl/expected.hpp>

#include <any>
#include <functional>

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

    HandleResult handle(const clang::Attr* attr,
                        const clang::Decl*,
                        const std::any* params,
                        SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
