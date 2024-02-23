#pragma once

#include "core/attribute_manager/backend_attribute_map.h"
#include "core/attribute_manager/common_attribute_map.h"
#include "core/attribute_manager/implicit_handlers/implicit_handler_map.h"

#include <clang/Sema/ParsedAttr.h>

#include <oklt/core/error.h>
#include <any>
#include <tl/expected.hpp>

#include <string>

namespace oklt {

struct Error;

class AttributeManager {
   protected:
    AttributeManager() = default;
    ~AttributeManager() = default;

   public:
    using AttrParamParserType =
        std::function<tl::expected<std::any, Error>(const clang::Attr*, SessionStage&)>;

    AttributeManager(const AttributeManager&) = delete;
    AttributeManager(AttributeManager&&) = delete;
    AttributeManager& operator=(const AttributeManager&) = delete;
    AttributeManager& operator=(AttributeManager&&) = delete;

    static AttributeManager& instance();

    template <typename AttrFrontendType>
    bool registerAttrFrontend(std::string name, AttrParamParserType paramParser) {
        static clang::ParsedAttrInfoRegistry::Add<AttrFrontendType> register_okl_atomic(name, "");
        auto [_, ret] = _attrParsers.try_emplace(std::move(name), std::move(paramParser));
        return ret;
    }

    bool registerCommonHandler(std::string name, AttrDeclHandler handler);
    bool registerCommonHandler(std::string name, AttrStmtHandler handler);

    bool registerBackendHandler(BackendAttributeMap::KeyType key, AttrDeclHandler handler);
    bool registerBackendHandler(BackendAttributeMap::KeyType key, AttrStmtHandler handler);

    bool registerImplicitHandler(ImplicitHandlerMap::KeyType key, DeclHandler handler);
    bool registerImplicitHandler(ImplicitHandlerMap::KeyType key, StmtHandler handler);

    tl::expected<std::any, Error> parseAttr(const clang::Attr* attr, SessionStage& stage);

    tl::expected<std::any, Error> handleAttr(const clang::Attr* attr,
                                             const clang::Decl* decl,
                                             const std::any& params,
                                             SessionStage& stage);
    tl::expected<std::any, Error> handleAttr(const clang::Attr* attr,
                                             const clang::Stmt* stmt,
                                             const std::any& params,
                                             SessionStage& stage);

    tl::expected<std::any, Error> handleDecl(const clang::Decl* decl, SessionStage& stage);
    tl::expected<std::any, Error> handleStmt(const clang::Stmt* stmt, SessionStage& stage);

    tl::expected<const clang::Attr*, Error> checkAttrs(const clang::AttrVec& attrs,
                                                       const clang::Decl* decl,
                                                       SessionStage& stage);
    tl::expected<const clang::Attr*, Error> checkAttrs(
        const clang::ArrayRef<const clang::Attr*>& attrs,
        const clang::Stmt* decl,
        SessionStage& stage);

   private:
    // INFO: here should not be the same named attributes in both
    //       might need to handle uniqueness

    // INFO: if build AttributeViwer just wrap into shared_ptr and copy it
    CommonAttributeMap _commonAttrs;
    BackendAttributeMap _backendAttrs;
    ImplicitHandlerMap _implicitHandlers;
    std::map<std::string, AttrParamParserType> _attrParsers;
};
}  // namespace oklt
