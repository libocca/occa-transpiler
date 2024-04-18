#pragma once

#include <oklt/core/error.h>
#include <oklt/util/string_utils.h>

#include "core/attribute_manager/handler_map.h"
#include "core/attribute_manager/result.h"

#include <clang/AST/ASTTypeTraits.h>
#include <clang/Sema/ParsedAttr.h>
#include <tl/expected.hpp>

#include <any>
#include <set>

namespace oklt {

struct OKLParsedAttr;

class AttributeManager {
   protected:
    AttributeManager() = default;
    ~AttributeManager() = default;

   public:
    AttributeManager(const AttributeManager&) = delete;
    AttributeManager(AttributeManager&&) = delete;
    AttributeManager& operator=(const AttributeManager&) = delete;
    AttributeManager& operator=(AttributeManager&&) = delete;

    static AttributeManager& instance();

    template <typename AttrFrontendType, typename F>
    bool registerAttrFrontend(std::string attr, F& func);
    template <typename F>
    bool registerCommonHandler(std::string attr, F& func);
    template <typename F>
    bool registerBackendHandler(TargetBackend, std::string attr, F& func);
    template <typename F>
    bool registerImplicitHandler(TargetBackend, F& func);
    template <typename F>
    bool registerSemaHandler(std::string attr, F& pre, F& post);

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
    [[nodiscard]] HandleResult handleSemaPost(SessionStage& stage,
                                              const clang::DynTypedNode& node,
                                              const clang::Attr* attr);

    tl::expected<std::set<const clang::Attr*>, Error> checkAttrs(SessionStage& stage,
                                                                 const clang::DynTypedNode& node);

   private:
    HandlerMap _handlers;
};

inline bool AttributeManager::hasImplicitHandler(TargetBackend backend, clang::ASTNodeKind kind) {
    return _handlers.hasHandler(backend, kind);
}

inline HandleResult AttributeManager::parseAttr(SessionStage& stage, const clang::Attr& attr) {
    return _handlers(stage, attr, nullptr);
}

inline HandleResult AttributeManager::parseAttr(SessionStage& stage,
                                                const clang::Attr& attr,
                                                OKLParsedAttr& params) {
    return _handlers(stage, attr, &params);
}

inline HandleResult AttributeManager::handleAttr(SessionStage& stage,
                                                 const clang::DynTypedNode& node,
                                                 const clang::Attr& attr,
                                                 const std::any* params) {
    return _handlers(stage, node, attr, params);
}

inline HandleResult AttributeManager::handleNode(SessionStage& stage,
                                                 const clang::DynTypedNode& node) {
    return _handlers(stage, node);
}

inline HandleResult AttributeManager::handleSemaPre(SessionStage& stage,
                                                    const clang::DynTypedNode& node,
                                                    const clang::Attr* attr) {
    return _handlers.pre(stage, node, attr);
}

inline HandleResult AttributeManager::handleSemaPost(SessionStage& stage,
                                                     const clang::DynTypedNode& node,
                                                     const clang::Attr* attr) {
    return _handlers.post(stage, node, attr);
}

}  // namespace oklt
