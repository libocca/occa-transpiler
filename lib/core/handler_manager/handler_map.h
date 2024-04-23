#pragma once

#include <oklt/core/target_backends.h>
#include "core/handler_manager/result.h"

#include <clang/AST/ASTTypeTraits.h>
#include <tl/expected.hpp>

#include <optional>
#include <string>
#include <tuple>

namespace oklt {

class SessionStage;
struct OKLParsedAttr;

enum HandleType {
    COMMON,
    BACKEND,
    IMPLICIT,
    SEMA,
    PARSER,
};

class NodeHandler {
    friend class HandlerMap;

   public:
    // Implicit
    virtual HandleResult handle(SessionStage&, const clang::DynTypedNode&) {
        return tl::make_unexpected("Unsupported handle call");
    }
    // Common & Backend
    virtual HandleResult handle(SessionStage&,
                                const clang::DynTypedNode&,
                                const clang::Attr&,
                                const std::any*) {
        return tl::make_unexpected("Unsupported handle call");
    }
    // Parser
    virtual HandleResult handle(SessionStage&, const clang::Attr&, OKLParsedAttr&) {
        return tl::make_unexpected("Unsupported handle call");
    }
    // Sema
    virtual HandleResult pre(SessionStage&, const clang::DynTypedNode&, const clang::Attr*) {
        return tl::make_unexpected("Unsupported handle call");
    }
    virtual HandleResult post(SessionStage&, const clang::DynTypedNode&, const clang::Attr*) {
        return tl::make_unexpected("Unsupported handle call");
    }

   protected:
    mutable clang::ASTNodeKind kind = {};
};

class HandlerMap;
struct HandleKeyBase {
    friend class HandlerMap;

    const HandleType t;
    std::optional<TargetBackend> backend;
    std::string attr = {};
    clang::ASTNodeKind kind = {};
    [[nodiscard]] auto key() const { return std::tie(t, backend, attr, kind); }

    bool operator<(const HandleKeyBase& rhs) const { return key() < rhs.key(); }

    explicit HandleKeyBase(HandleType k)
        : t(k){};
};

template <enum HandleType H, typename E = void>
struct HandlerKey;

class HandlerMap {
   public:
    HandlerMap() = default;
    ~HandlerMap() = default;

    template <enum HandleType H, typename T>
    bool insert(HandlerKey<H>&& key, T& func);
    template <enum HandleType H, typename T>
    bool insert(HandlerKey<H>&& key, T& pre, T& post);

    // Implicit
    [[nodiscard]] bool hasHandler(TargetBackend, clang::ASTNodeKind) const;
    HandleResult operator()(SessionStage&, const clang::DynTypedNode&);

    // Common & Backend
    [[nodiscard]] bool hasHandler(const std::string&, clang::ASTNodeKind) const;
    [[nodiscard]] bool hasHandler(TargetBackend, const std::string&, clang::ASTNodeKind) const;
    HandleResult operator()(SessionStage&,
                            const clang::DynTypedNode&,
                            const clang::Attr&,
                            const std::any* params);

    // Parser
    [[nodiscard]] bool hasHandler(const std::string&) const;
    HandleResult operator()(SessionStage& stage, const clang::Attr& attr, OKLParsedAttr* params);

    // Sema
    [[nodiscard]] bool hasSemeHandler(const std::string&, clang::ASTNodeKind) const;
    HandleResult pre(SessionStage&, const clang::DynTypedNode&, const clang::Attr*);
    HandleResult post(SessionStage&, const clang::DynTypedNode&, const clang::Attr*);

   private:
    std::map<HandleKeyBase, std::unique_ptr<NodeHandler>> _nodeHandlers;
};

template <enum HandleType H, typename T>
inline bool HandlerMap::insert(HandlerKey<H>&& key, T& func) {
    using NodeHandleType = typename HandlerKey<H>::HandlerType;
    auto handler = std::unique_ptr<NodeHandler>(new NodeHandleType(func));
    key.kind = handler->kind;
    return _nodeHandlers.try_emplace(std::move(key), std::move(handler)).second;
}

template <enum HandleType H, typename T>
inline bool HandlerMap::insert(HandlerKey<H>&& key, T& pre, T& post) {
    using NodeHandleType = typename HandlerKey<H>::HandlerType;
    auto handler = std::unique_ptr<NodeHandler>(new NodeHandleType(pre, post));
    key.kind = handler->kind;
    return _nodeHandlers.try_emplace(std::move(key), std::move(handler)).second;
}

}  // namespace oklt
