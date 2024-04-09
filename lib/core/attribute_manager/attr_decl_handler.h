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
    using HandleType = std::function<
        HandleResult(SessionStage&, const clang::Decl&, const clang::Attr&, const std::any*)>;

    explicit AttrDeclHandler(HandleType h)
        : _handler(std::move(h)) {}

    AttrDeclHandler(AttrDeclHandler&&) = default;
    ~AttrDeclHandler() = default;

    HandleResult handle(SessionStage& stage,
                        const clang::Decl&,
                        const clang::Attr& attr,
                        const std::any* params);

   private:
    HandleType _handler;
};
}  // namespace oklt
