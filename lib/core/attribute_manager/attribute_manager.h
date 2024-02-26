#pragma once

#include <oklt/core/error.h>
#include <oklt/util/string_utils.h>
#include "core/attribute_manager/backend_attribute_map.h"
#include "core/attribute_manager/common_attribute_map.h"
#include "core/attribute_manager/implicit_handlers/implicit_handler_map.h"
#include "core/attribute_manager/result.h"

#include <clang/Sema/ParsedAttr.h>
#include <any>
#include <tl/expected.hpp>

#include <string>
#include <type_traits>

namespace oklt {

struct Error;

class AttributeManager {
   protected:
    AttributeManager() = default;
    ~AttributeManager() = default;

   public:
    using AttrParamParserType = std::function<ParseResult(const clang::Attr*, SessionStage&)>;

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

    ParseResult parseAttr(const clang::Attr* attr, SessionStage& stage);

    HandleResult handleAttr(const clang::Attr* attr,
                            const clang::Decl* decl,
                            const std::any* params,
                            SessionStage& stage);
    HandleResult handleAttr(const clang::Attr* attr,
                            const clang::Stmt* stmt,
                            const std::any* params,
                            SessionStage& stage);

    HandleResult handleDecl(const clang::Decl* decl, SessionStage& stage);
    HandleResult handleStmt(const clang::Stmt* stmt, SessionStage& stage);

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

namespace detail {
template <typename Handler, typename DeclStmt, typename AttrHandler>
AttrHandler makeSpecificAttrXXXHandle(Handler& handler) {
    using ParamsType = typename std::remove_pointer_t<typename func_param_type<Handler, 3>::type>;
    using HandleDeclStmt =
        typename std::remove_pointer_t<typename func_param_type<Handler, 2>::type>;
    constexpr size_t n_arguments = func_num_arguments<Handler>::value;

    return AttrHandler{[&handler, n_arguments](const clang::Attr* attr,
                                               const DeclStmt* declStmt,
                                               const std::any* params,
                                               SessionStage& stage) -> HandleResult {
        static_assert(
            n_arguments == N_ARGUMENTS_WITH_PARAMS || n_arguments == N_ARGUMENTS_WITHOUT_PARAMS,
            "Handler must have 3 or 4 arguments");
        const auto* handleDeclStmt = clang::dyn_cast_or_null<HandleDeclStmt>(declStmt);
        if (handleDeclStmt == nullptr) {
            auto baseDeclStmtTypeName = typeid(DeclStmt).name();
            auto handleDeclStmtTypeName = typeid(HandleDeclStmt).name();
            return tl::make_unexpected(Error{
                {},
                util::fmt("Failed to cast {} to {}", baseDeclStmtTypeName, handleDeclStmtTypeName)
                    .value()});
        }
        if constexpr (n_arguments == N_ARGUMENTS_WITH_PARAMS) {
            const ParamsType* params_ptr = std::any_cast<ParamsType>(params);
            if (params_ptr == nullptr) {
                return tl::make_unexpected(Error{
                    {},
                    util::fmt("Any cast fail: failed to cast to {}", typeid(ParamsType).name())
                        .value()});
            }
            return handler(attr, handleDeclStmt, params_ptr, stage);
        } else {
            return handler(attr, handleDeclStmt, stage);
        }
    }};
}
}  // namespace detail

template <typename Handler>
auto makeSpecificAttrHandle(Handler& handler) {
    using DeclOrStmt = typename std::remove_const_t<
        typename std::remove_pointer_t<typename func_param_type<Handler, 2>::type>>;

    // if constexpr (std::is_same_v<DeclOrStmt, clang::Decl>) {
    if constexpr (std::is_base_of_v<clang::Decl, DeclOrStmt>) {
        return detail::makeSpecificAttrXXXHandle<Handler, clang::Decl, AttrDeclHandler>(handler);
    } else {
        return detail::makeSpecificAttrXXXHandle<Handler, clang::Stmt, AttrStmtHandler>(handler);
    }
}

}  // namespace oklt
