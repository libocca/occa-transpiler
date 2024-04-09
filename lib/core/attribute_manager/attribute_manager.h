#pragma once

#include <oklt/core/error.h>
#include <oklt/util/string_utils.h>

#include "core/attribute_manager/backend_attribute_map.h"
#include "core/attribute_manager/common_attribute_map.h"
#include "core/attribute_manager/implicit_handlers/implicit_handler_map.h"
#include "core/attribute_manager/result.h"

#include <clang/Sema/ParsedAttr.h>
#include <tl/expected.hpp>

#include <any>
#include <set>
#include <type_traits>

namespace oklt {

struct Error;
struct OKLParsedAttr;

class AttributeManager {
   protected:
    AttributeManager() = default;
    ~AttributeManager() = default;

   public:
    using AttrParamParserType =
        std::function<ParseResult(SessionStage&, const clang::Attr&, OKLParsedAttr&)>;

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

    ParseResult parseAttr(SessionStage& stage, const clang::Attr& attr);
    ParseResult parseAttr(SessionStage& stage, const clang::Attr& attr, OKLParsedAttr& params);

    bool hasImplicitHandler(TargetBackend backend, int nodeType) {
        return _implicitHandlers.hasHandler({backend, nodeType});
    }

    HandleResult handleAttr(SessionStage& stage,
                            const clang::Decl& decl,
                            const clang::Attr& attr,
                            const std::any* params);
    HandleResult handleAttr(SessionStage& stage,
                            const clang::Stmt& stmt,
                            const clang::Attr& attr,
                            const std::any* params);

    HandleResult handleNode(SessionStage& stage, const clang::Decl& decl);
    HandleResult handleNode(SessionStage& stage, const clang::Stmt& stmt);

    tl::expected<std::set<const clang::Attr*>, Error> checkAttrs(SessionStage& stage,
                                                                 const clang::Decl& decl);
    tl::expected<std::set<const clang::Attr*>, Error> checkAttrs(SessionStage& stage,
                                                                 const clang::Stmt& stmt);

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
template <typename Handler, typename NodeType, typename AttrHandler>
AttrHandler makeSpecificAttrXXXHandle(Handler& handler) {
    using HandleDeclStmt =
        typename std::remove_reference_t<typename func_param_type<Handler, 2>::type>;
    constexpr size_t n_arguments = func_num_arguments<Handler>::value;

    return AttrHandler{[&handler, n_arguments](SessionStage& stage,
                                               const NodeType& node,
                                               const clang::Attr& attr,
                                               const std::any* params) -> HandleResult {
        static_assert(
            n_arguments == N_ARGUMENTS_WITH_PARAMS || n_arguments == N_ARGUMENTS_WITHOUT_PARAMS,
            "Handler must have 3 or 4 arguments");
        const auto localNode = clang::dyn_cast_or_null<HandleDeclStmt>(&node);
        if (!localNode) {
            auto baseNodeTypeName = typeid(NodeType).name();
            auto handleNodeTypeName = typeid(HandleDeclStmt).name();
            return tl::make_unexpected(
                Error{{},
                      util::fmt("Failed to cast {} to {}", baseNodeTypeName, handleNodeTypeName)
                          .value()});
        }
        if constexpr (n_arguments == N_ARGUMENTS_WITH_PARAMS) {
            using ParamsType =
                typename std::remove_pointer_t<typename func_param_type<Handler, 4>::type>;
            const ParamsType* params_ptr =
                params->type() == typeid(ParamsType) ? std::any_cast<ParamsType>(params) : nullptr;
            if (!params_ptr) {
                return tl::make_unexpected(Error{
                    {},
                    util::fmt("Any cast fail: failed to cast to {}", typeid(ParamsType).name())
                        .value()});
            }
            return handler(stage, *localNode, attr, params_ptr);
        } else {
            return handler(stage, *localNode, attr);
        }
    }};
}

template <typename Handler, typename NodeType, typename NodeHandler>
NodeHandler makeSpecificImplicitXXXHandle(Handler& handler) {
    using HandleDeclStmt =
        typename std::remove_reference_t<typename func_param_type<Handler, 2>::type>;

    return NodeHandler{[&handler](SessionStage& stage, const NodeType& node) -> HandleResult {
        const auto localNode = clang::dyn_cast_or_null<HandleDeclStmt>(&node);
        if (!localNode) {
            auto baseNodeTypeName = typeid(NodeType).name();
            auto handleNodeTypeName = typeid(HandleDeclStmt).name();
            return tl::make_unexpected(
                Error{{},
                      util::fmt("Failed to cast {} to {}", baseNodeTypeName, handleNodeTypeName)
                          .value()});
        }
        return handler(stage, *localNode);
    }};
}
}  // namespace detail

template <typename Handler>
auto makeSpecificAttrHandle(Handler& handler) {
    using DeclOrStmt = typename std::remove_const_t<
        typename std::remove_reference_t<typename func_param_type<Handler, 2>::type>>;

    if constexpr (std::is_base_of_v<clang::Decl, DeclOrStmt>) {
        return detail::makeSpecificAttrXXXHandle<Handler, clang::Decl, AttrDeclHandler>(handler);
    } else {
        return detail::makeSpecificAttrXXXHandle<Handler, clang::Stmt, AttrStmtHandler>(handler);
    }
}

template <typename Handler>
auto makeSpecificImplicitHandle(Handler& handler) {
    using nodeType = typename std::remove_const_t<
        typename std::remove_reference_t<typename func_param_type<Handler, 2>::type>>;

    if constexpr (std::is_base_of_v<clang::Decl, nodeType>) {
        return detail::makeSpecificImplicitXXXHandle<Handler, clang::Decl, DeclHandler>(handler);
    } else {
        return detail::makeSpecificImplicitXXXHandle<Handler, clang::Stmt, StmtHandler>(handler);
    }
}

}  // namespace oklt
