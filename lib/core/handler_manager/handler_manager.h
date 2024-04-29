#pragma once

#include <oklt/core/error.h>
#include "util/string_utils.hpp"

#include "core/handler_manager/handler_map.h"
#include "core/handler_manager/result.h"

#include <clang/AST/ASTTypeTraits.h>
#include <clang/Sema/ParsedAttr.h>
#include <tl/expected.hpp>

#include <any>
#include <set>

namespace oklt {

struct OKLParsedAttr;

class HandlerManager {
   public:
    HandlerManager() = default;
    ~HandlerManager() = default;

    template <typename AttrFrontendType, typename F>
    friend bool registerAttrFrontend(std::string attr, F& func);
    template <typename F>
    friend bool registerCommonHandler(std::string attr, F& func);
    template <typename F>
    friend bool registerBackendHandler(TargetBackend, std::string attr, F& func);
    template <typename F>
    friend bool registerImplicitHandler(TargetBackend, F& func);
    template <typename F>
    friend bool registerSemaHandler(std::string attr, F& pre, F& post);

    [[nodiscard]] bool hasImplicitHandler(TargetBackend backend, clang::ASTNodeKind kind);
    [[nodiscard]] HandleResult parseAttr(SessionStage& stage, const clang::Attr& attr);
    [[nodiscard]] HandleResult parseAttr(SessionStage& stage,
                                         const clang::Attr& attr,
                                         OKLParsedAttr& params);

    [[nodiscard]] HandleResult handleAttr(SessionStage& stage,
                                          const clang::DynTypedNode& node,
                                          const clang::Attr& attr,
                                          const std::any* params);
    [[nodiscard]] HandleResult handleNode(SessionStage& stage, const clang::DynTypedNode& node);

    [[nodiscard]] HandleResult handleSemaPre(SessionStage& stage,
                                             const clang::DynTypedNode& node,
                                             const clang::Attr* attr);
    [[nodiscard]] static HandleResult handleSemaPost(SessionStage& stage,
                                                     const clang::DynTypedNode& node,
                                                     const clang::Attr* attr);

    tl::expected<std::set<const clang::Attr*>, Error> checkAttrs(SessionStage& stage,
                                                                 const clang::DynTypedNode& node);

   private:
    static HandlerMap& _map();
};

inline bool HandlerManager::hasImplicitHandler(TargetBackend backend, clang::ASTNodeKind kind) {
    return _map().hasHandler(backend, kind);
}

inline HandleResult HandlerManager::parseAttr(SessionStage& stage, const clang::Attr& attr) {
    return _map()(stage, attr, nullptr);
}

inline HandleResult HandlerManager::parseAttr(SessionStage& stage,
                                              const clang::Attr& attr,
                                              OKLParsedAttr& params) {
    return _map()(stage, attr, &params);
}

inline HandleResult HandlerManager::handleAttr(SessionStage& stage,
                                               const clang::DynTypedNode& node,
                                               const clang::Attr& attr,
                                               const std::any* params) {
    return _map()(stage, node, attr, params);
}

inline HandleResult HandlerManager::handleNode(SessionStage& stage,
                                               const clang::DynTypedNode& node) {
    return _map()(stage, node);
}

inline HandleResult HandlerManager::handleSemaPre(SessionStage& stage,
                                                  const clang::DynTypedNode& node,
                                                  const clang::Attr* attr) {
    return _map().pre(stage, node, attr);
}

inline HandleResult HandlerManager::handleSemaPost(SessionStage& stage,
                                                   const clang::DynTypedNode& node,
                                                   const clang::Attr* attr) {
    return _map().post(stage, node, attr);
}

}  // namespace oklt
