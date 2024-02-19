#pragma once

#include <clang/AST/Attr.h>

#include <functional>

namespace oklt {

class SessionStage;

class AttrDeclHandler {
   public:
    using HandleType = std::function<bool(const clang::Attr*, const clang::Decl*, SessionStage&)>;

    explicit AttrDeclHandler(HandleType h)
        : _handler(std::move(h)) {}

    AttrDeclHandler(AttrDeclHandler&&) = default;
    ~AttrDeclHandler() = default;

    bool handle(const clang::Attr* attr, const clang::Decl*, SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
