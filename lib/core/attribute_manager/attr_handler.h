#pragma once

#include "core/attribute_manager/result.h"

#include <clang/AST/ASTTypeTraits.h>
#include <clang/AST/Attr.h>
#include <tl/expected.hpp>

#include <any>
#include <functional>

namespace oklt {

class SessionStage;

class AttrHandler {
   public:
    using HandleType = std::function<HandleResult(SessionStage&,
                                                  const clang::DynTypedNode&,
                                                  const clang::Attr&,
                                                  const std::any*)>;

    explicit AttrHandler(HandleType h)
        : _handler(std::move(h)) {}

    AttrHandler(AttrHandler&&) = default;
    ~AttrHandler() = default;

    HandleResult handle(SessionStage& stage,
                        const clang::DynTypedNode&,
                        const clang::Attr& attr,
                        const std::any* params);

   private:
    HandleType _handler;
};
}  // namespace oklt
