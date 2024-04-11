#include <oklt/core/error.h>

#include "attributes/frontend/params/empty_params.h"
#include "attributes/utils/parser.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt {
using namespace clang;

AttributeManager& AttributeManager::instance() {
    static AttributeManager attrManager;
    return attrManager;
}

bool AttributeManager::registerCommonHandler(CommonAttributeMap::KeyType key, AttrHandler handler) {
    return _commonAttrs.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerBackendHandler(BackendAttributeMap::KeyType key,
                                              AttrHandler handler) {
    return _backendAttrs.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerImplicitHandler(ImplicitHandlerMap::KeyType key,
                                               NodeHandler handler) {
    return _implicitHandlers.registerHandler(std::move(key), std::move(handler));
}

HandleResult AttributeManager::handleNode(SessionStage& stage, const DynTypedNode& node) {
    return _implicitHandlers(stage, node);
}

HandleResult AttributeManager::handleAttr(SessionStage& stage,
                                          const DynTypedNode& node,
                                          const Attr& attr,
                                          const std::any* params) {
    auto name = attr.getNormalizedFullName();
    auto kind = node.getNodeKind();
    if (_commonAttrs.hasHandler({name, kind})) {
        return _commonAttrs.handleAttr(stage, node, attr, params);
    }

    auto backend = stage.getBackend();
    if (_backendAttrs.hasHandler({backend, name, kind})) {
        return _backendAttrs.handleAttr(stage, node, attr, params);
    }

    // TODO: Uncomment after multi-handle added
    // return tl::make_unexpected(Error{std::error_code(), "no handler for attr: " + name});
    return {};
}

HandleResult AttributeManager::parseAttr(SessionStage& stage, const Attr& attr) {
    std::string name = attr.getNormalizedFullName();
    auto it = _attrParsers.find(name);
    if (it != _attrParsers.end()) {
        auto params = ParseOKLAttr(stage, attr);
        return it->second(stage, attr, params);
    }

    return EmptyParams{};
}

HandleResult AttributeManager::parseAttr(SessionStage& stage,
                                         const Attr& attr,
                                        OKLParsedAttr& params) {
    auto it = _attrParsers.find(params.name);
    if (it != _attrParsers.end()) {
        return it->second(stage, attr, params);
    }
    return EmptyParams{};
}

tl::expected<std::set<const Attr*>, Error> AttributeManager::checkAttrs(SessionStage& stage,
                                                                        const DynTypedNode& node) {
    auto backend = stage.getBackend();
    auto kind = node.getNodeKind();
    if (ASTNodeKind::getFromNodeKind<Decl>().isBaseOf(kind)) {
        auto& decl = node.getUnchecked<Decl>();
        if (!decl.hasAttrs()) {
            return {};
        }

        const auto& attrs = decl.getAttrs();

        // in case of multiple same attribute take the last one
        std::set<const Attr*> collectedAttrs;
        for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
            const auto& attr = *it;

            if (!attr || !isOklAttribute(*attr)) {
                continue;
            }

            auto name = attr->getNormalizedFullName();
            if (!_commonAttrs.hasHandler({name, kind}) &&
                !_backendAttrs.hasHandler({backend, name, kind})) {
                // TODO report diag error
                SPDLOG_ERROR(
                    "{} attribute: {} for decl: {} does not have a registered handler",
                    decl.getBeginLoc().printToString(decl.getASTContext().getSourceManager()),
                    name,
                    decl.getDeclKindName());

                return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "no handler"});
            }

            auto [_, isNew] = collectedAttrs.insert(attr);
            if (!isNew) {
                // TODO convince OCCA community to specify such case as forbidden
                SPDLOG_ERROR(
                    "{} multi declaration of attribute: {} for decl: {}",
                    decl.getBeginLoc().printToString(decl.getASTContext().getSourceManager()),
                    name,
                    decl.getDeclKindName());
            }
        }

        return collectedAttrs;
    }

    if (ASTNodeKind::getFromNodeKind<AttributedStmt>().isBaseOf(kind)) {
        auto& stmt = node.getUnchecked<AttributedStmt>();
        auto subKind = ASTNodeKind::getFromNode(*stmt.getSubStmt());

        const auto& attrs = cast<AttributedStmt>(stmt).getAttrs();
        std::set<const Attr*> collectedAttrs;
        for (const auto attr : attrs) {
            if (!attr || !isOklAttribute(*attr)) {
                continue;
            }

            auto name = attr->getNormalizedFullName();
            if (!_commonAttrs.hasHandler({name, subKind}) &&
                !_backendAttrs.hasHandler({backend, name, subKind})) {
                // TODO report diag error
                SPDLOG_ERROR(
                    "{} attribute: {} for stmt: {} does not have a registered handler",
                    stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager()),
                    name,
                    stmt.getStmtClassName());
                return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "no handler"});
            }

            auto [_, isNew] = collectedAttrs.insert(attr);
            if (!isNew) {
                // TODO convince OCCA community to specify such case as forbidden
                SPDLOG_ERROR(
                    "{} multi declaration of attribute: {} for stmt: {}",
                    stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager()),
                    name,
                    stmt.getStmtClassName());
            }
        }

        return collectedAttrs;
    }

    return {};
}

}  // namespace oklt
