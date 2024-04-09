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

bool AttributeManager::registerCommonHandler(std::string name, AttrDeclHandler handler) {
    return _commonAttrs.registerHandler(std::move(name), std::move(handler));
}

bool AttributeManager::registerCommonHandler(std::string name, AttrStmtHandler handler) {
    return _commonAttrs.registerHandler(std::move(name), std::move(handler));
}

bool AttributeManager::registerBackendHandler(BackendAttributeMap::KeyType key,
                                              AttrDeclHandler handler) {
    return _backendAttrs.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerBackendHandler(BackendAttributeMap::KeyType key,
                                              AttrStmtHandler handler) {
    return _backendAttrs.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerImplicitHandler(ImplicitHandlerMap::KeyType key,
                                               DeclHandler handler) {
    return _implicitHandlers.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerImplicitHandler(ImplicitHandlerMap::KeyType key,
                                               StmtHandler handler) {
    return _implicitHandlers.registerHandler(std::move(key), std::move(handler));
}

HandleResult AttributeManager::handleNode(SessionStage& stage, const Stmt& stmt) {
    return _implicitHandlers(stage, stmt);
}

HandleResult AttributeManager::handleNode(SessionStage& stage, const Decl& decl) {
    return _implicitHandlers(stage, decl);
}

HandleResult AttributeManager::handleAttr(SessionStage& stage,
                                          const Decl& decl,
                                          const Attr& attr,
                                          const std::any* params) {
    std::string name = attr.getNormalizedFullName();
    if (_commonAttrs.hasAttrHandler(name)) {
        return _commonAttrs.handleAttr(stage, decl, attr, params);
    }

    if (_backendAttrs.hasAttrHandler(stage, name)) {
        return _backendAttrs.handleAttr(stage, decl, attr, params);
    }

    // TODO: Uncomment after multi-handle added
    // return tl::make_unexpected(Error{std::error_code(), "no handler for attr: " + name});
    return {};
}

HandleResult AttributeManager::handleAttr(SessionStage& stage,
                                          const Stmt& stmt,
                                          const Attr& attr,
                                          const std::any* params) {
    std::string name = attr.getNormalizedFullName();
    if (_commonAttrs.hasAttrHandler(name)) {
        return _commonAttrs.handleAttr(stage, stmt, attr, params);
    }

    if (_backendAttrs.hasAttrHandler(stage, name)) {
        return _backendAttrs.handleAttr(stage, stmt, attr, params);
    }

    // TODO: Uncomment after multi-handle added
    // return tl::make_unexpected(Error{std::error_code(), "no handler for attr: " + name});
    return {};
}

ParseResult AttributeManager::parseAttr(SessionStage& stage, const Attr& attr) {
    std::string name = attr.getNormalizedFullName();
    auto it = _attrParsers.find(name);
    if (it != _attrParsers.end()) {
        auto params = ParseOKLAttr(stage, attr);
        return it->second(stage, attr, params);
    }

    return EmptyParams{};
}

ParseResult AttributeManager::parseAttr(SessionStage& stage,
                                        const Attr& attr,
                                        OKLParsedAttr& params) {
    auto it = _attrParsers.find(params.name);
    if (it != _attrParsers.end()) {
        return it->second(stage, attr, params);
    }
    return EmptyParams{};
}

tl::expected<std::set<const Attr*>, Error> AttributeManager::checkAttrs(SessionStage& stage,
                                                                        const Decl& decl) {
    if (!decl.hasAttrs()) {
        return {};
    }

    const auto& attrs = decl.getAttrs();
    std::set<const Attr*> collectedAttrs;

    // in case of multiple same attribute take the last one
    for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
        const auto& attr = *it;

        if (!attr) {
            continue;
        }

        if (!isOklAttribute(*attr)) {
            continue;
        }

        auto name = attr->getNormalizedFullName();
        if (!_commonAttrs.hasAttrHandler(name) && !_backendAttrs.hasAttrHandler(stage, name)) {
            // TODO report diag error
            SPDLOG_ERROR("{} attribute: {} for decl: {} doesn not have a registered handler",
                         decl.getBeginLoc().printToString(decl.getASTContext().getSourceManager()),
                         name,
                         decl.getDeclKindName());

            return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "no handler"});
        }

        auto [_, isNew] = collectedAttrs.insert(attr);
        if (!isNew) {
            // TODO convince OCCA community to specify such case as forbidden
            SPDLOG_ERROR("{} multi declaration of attribute: {} for decl: {}",
                         decl.getBeginLoc().printToString(decl.getASTContext().getSourceManager()),
                         name,
                         decl.getDeclKindName());
        }
    }

    return collectedAttrs;
}

tl::expected<std::set<const Attr*>, Error> AttributeManager::checkAttrs(SessionStage& stage,
                                                                        const Stmt& stmt) {
    if (stmt.getStmtClass() != Stmt::AttributedStmtClass) {
        return {};
    }

    const auto& attrs = cast<AttributedStmt>(stmt).getAttrs();
    std::set<const Attr*> collectedAttrs;
    for (const auto attr : attrs) {
        if (!attr) {
            continue;
        }

        if (!isOklAttribute(*attr)) {
            continue;
        }

        auto name = attr->getNormalizedFullName();
        if (!_commonAttrs.hasAttrHandler(name) && !_backendAttrs.hasAttrHandler(stage, name)) {
            // TODO report diag error
            SPDLOG_ERROR("{} attribute: {} for stmt: {} does not have a registered handler",
                         stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager()),
                         name,
                         stmt.getStmtClassName());
            return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "no handler"});
        }

        auto [_, isNew] = collectedAttrs.insert(attr);
        if (!isNew) {
            // TODO convince OCCA community to specify such case as forbidden
            SPDLOG_ERROR("{} multi declaration of attribute: {} for stmt: {}",
                         stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager()),
                         name,
                         stmt.getStmtClassName());
        }
    }

    return collectedAttrs;
}
}  // namespace oklt
