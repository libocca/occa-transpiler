#pragma once

#include <oklt/core/error.h>
#include <oklt/util/string_utils.h>

#include "core/attribute_manager/backend_attribute_map.h"
#include "core/attribute_manager/common_attribute_map.h"
#include "core/attribute_manager/implicit_handlers/implicit_handler_map.h"
#include "core/attribute_manager/result.h"

#include <clang/AST/ASTTypeTraits.h>
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
        std::function<HandleResult(SessionStage&, const clang::Attr&, OKLParsedAttr&)>;

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

    bool registerCommonHandler(CommonAttributeMap::KeyType key, AttrHandler handler);
    bool registerBackendHandler(BackendAttributeMap::KeyType key, AttrHandler handler);
    bool registerImplicitHandler(ImplicitHandlerMap::KeyType key, NodeHandler handler);

    HandleResult parseAttr(SessionStage& stage, const clang::Attr& attr);
    HandleResult parseAttr(SessionStage& stage, const clang::Attr& attr, OKLParsedAttr& params);

    bool hasImplicitHandler(TargetBackend backend, clang::ASTNodeKind kind) {
        return _implicitHandlers.hasHandler({backend, kind});
    }

    HandleResult handleAttr(SessionStage& stage,
                            const clang::DynTypedNode& node,
                            const clang::Attr& attr,
                            const std::any* params);

    HandleResult handleNode(SessionStage& stage, const clang::DynTypedNode& node);

    tl::expected<std::set<const clang::Attr*>, Error> checkAttrs(SessionStage& stage,
                                                                 const clang::DynTypedNode& node);

   private:
    // INFO: here should not be the same named attributes in both
    //       might need to handle uniqueness

    // INFO: if build AttributeViewer just wrap into shared_ptr and copy it
    CommonAttributeMap _commonAttrs;
    BackendAttributeMap _backendAttrs;
    ImplicitHandlerMap _implicitHandlers;
    std::map<std::string, AttrParamParserType> _attrParsers;
};

namespace detail {
template <typename Handler, typename AttrHandler>
AttrHandler makeSpecificAttrXXXHandle(Handler& handler) {
    using NodeType = typename std::remove_reference_t<typename func_param_type<Handler, 2>::type>;
    constexpr size_t nargs = func_num_arguments<Handler>::value;

    return AttrHandler{[&handler, nargs](SessionStage& stage,
                                         const clang::DynTypedNode& node,
                                         const clang::Attr& attr,
                                               const std::any* params) -> HandleResult {
        static_assert(nargs == N_ARGUMENTS_WITH_PARAMS || nargs == N_ARGUMENTS_WITHOUT_PARAMS,
                      "Handler must have 3 or 4 arguments");

        const auto localNode = node.get<NodeType>();
        if (!localNode) {
            auto baseNodeTypeName = node.getNodeKind().asStringRef();
            auto handleNodeTypeName = clang::ASTNodeKind::getFromNodeKind<NodeType>().asStringRef();
            return tl::make_unexpected(Error{
                {},
                util::fmt(
                    "Failed to cast {} to {}", baseNodeTypeName.str(), handleNodeTypeName.str())
                    .value()});
        }

        if constexpr (nargs == N_ARGUMENTS_WITHOUT_PARAMS) {
            return handler(stage, *localNode, attr);
        } else {
            using ParamsType =
                typename std::remove_pointer_t<typename func_param_type<Handler, 4>::type>;
            if constexpr (std::is_same_v<std::remove_const_t<ParamsType>, std::any>) {
                return handler(stage, *localNode, attr, params);
            } else {
                const ParamsType* params_ptr = params->type() == typeid(ParamsType)
                                                   ? std::any_cast<ParamsType>(params)
                                                   : nullptr;
                if (!params_ptr) {
                    return tl::make_unexpected(Error{
                        {},
                        util::fmt("Any cast fail: failed to cast to {}", typeid(ParamsType).name())
                            .value()});
                }
                return handler(stage, *localNode, attr, params_ptr);
            }
        }
    }};
}

template <typename Handler, typename NodeHandler>
NodeHandler makeSpecificImplicitXXXHandle(Handler& handler) {
    using NodeType = typename std::remove_reference_t<typename func_param_type<Handler, 2>::type>;

    return NodeHandler{[&handler](SessionStage& stage,
                                  const clang::DynTypedNode& node) -> HandleResult {
        const auto localNode = node.get<NodeType>();
        if (!localNode) {
            auto baseNodeTypeName = node.getNodeKind().asStringRef();
            auto handleNodeTypeName = clang::ASTNodeKind::getFromNodeKind<NodeType>().asStringRef();
            return tl::make_unexpected(Error{
                {},
                util::fmt(
                    "Failed to cast {} to {}", baseNodeTypeName.str(), handleNodeTypeName.str())
                    .value()});
        }

        return handler(stage, *localNode);
    }};
}
}  // namespace detail

template <typename Handler>
auto makeSpecificAttrHandle(Handler& handler) {
    return detail::makeSpecificAttrXXXHandle<Handler, AttrHandler>(handler);
}

template <typename Handler>
auto makeSpecificImplicitHandle(Handler& handler) {
    return detail::makeSpecificImplicitXXXHandle<Handler, NodeHandler>(handler);
}

}  // namespace oklt
