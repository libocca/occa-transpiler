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

    HandleResult parseAttr(SessionStage& stage, const clang::Attr& attr);
    HandleResult parseAttr(SessionStage& stage, const clang::Attr& attr, OKLParsedAttr& params);
    HandleResult handleAttr(SessionStage& stage,
                            const clang::DynTypedNode& node,
                            const clang::Attr& attr,
                            const std::any* params);

    HandleResult handleNode(SessionStage& stage, const clang::DynTypedNode& node);

    tl::expected<std::set<const clang::Attr*>, Error> checkAttrs(SessionStage& stage,
                                                                 const clang::DynTypedNode& node);

   private:
    HandlerMap _handlers;
};

}  // namespace oklt
